"""Gradio WebUI for batch image segmentation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from img_seg.config import PROJECT_ROOT
from img_seg.inference import (
    DEFAULT_OUTPUT_DIR,
    InferenceCancelledError,
    collect_inputs,
    collect_reference_masks,
    evaluate_result_metrics,
    list_available_models,
    list_checkpoints,
    make_preview,
    run_batch_inference,
)

APP_TITLE = "ImgSeg WebUI"
DEFAULT_CONFIG = "configs/nnunet_v2/base.yaml"
WEBUI_PROGRESS = gr.Progress(track_tqdm=True)
WEBUI_CANCEL_EVENTS: dict[str, threading.Event] = {}
WEBUI_CANCEL_LOCK = threading.Lock()

CSS = """
:root {
  --imgseg-bg: #f8fafc;
  --imgseg-panel: #ffffff;
  --imgseg-text: #0f172a;
  --imgseg-muted: #475569;
  --imgseg-border: #e2e8f0;
  --imgseg-teal: #0f766e;
  --imgseg-blue: #2563eb;
  --imgseg-warn: #f59e0b;
}
html,
body,
gradio-app,
.gradio-container {
  min-height: 100%;
  background: var(--imgseg-bg) !important;
}
.gradio-container {
  background: var(--imgseg-bg);
  color: var(--imgseg-text);
  max-width: 1440px !important;
}
#imgseg-shell {
  gap: 16px;
}
#imgseg-header {
  border-bottom: 1px solid var(--imgseg-border);
  padding: 10px 0 14px 0;
  margin-bottom: 6px;
}
#imgseg-header h1 {
  font-size: 24px;
  line-height: 1.2;
  margin: 0;
  color: var(--imgseg-text);
}
#imgseg-header p {
  margin: 6px 0 0 0;
  color: var(--imgseg-muted);
  font-size: 13px;
}
#imgseg-run {
  background: var(--imgseg-teal);
  border-color: var(--imgseg-teal);
}
#imgseg-stop {
  border-color: #dc2626;
  color: #dc2626;
}
#imgseg-preview-button {
  min-height: 72px;
}
.imgseg-status textarea {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --imgseg-bg: #020617;
    --imgseg-panel: #0f172a;
    --imgseg-text: #e2e8f0;
    --imgseg-muted: #94a3b8;
    --imgseg-border: #1e293b;
    --imgseg-teal: #14b8a6;
    --imgseg-blue: #60a5fa;
    --imgseg-warn: #fbbf24;
  }
}
.dark,
.dark body,
.dark gradio-app,
.dark .gradio-container,
[data-theme="dark"],
[data-theme="dark"] body,
[data-theme="dark"] gradio-app,
[data-theme="dark"] .gradio-container {
  background: #020617 !important;
  color: #e2e8f0 !important;
}
.dark #imgseg-header,
[data-theme="dark"] #imgseg-header {
  border-bottom-color: #1e293b;
}
.dark #imgseg-header h1,
[data-theme="dark"] #imgseg-header h1 {
  color: #e2e8f0;
}
.dark #imgseg-header p,
[data-theme="dark"] #imgseg-header p {
  color: #94a3b8;
}
"""


def _model_choices() -> list[tuple[str, str]]:
    return [(model.label, model.key) for model in list_available_models() if model.runnable]


def _checkpoint_choices() -> list[tuple[str, str]]:
    return [(checkpoint.label, str(checkpoint.path)) for checkpoint in list_checkpoints()]


def _default_checkpoint() -> str:
    checkpoints = _checkpoint_choices()
    return checkpoints[0][1] if checkpoints else ""


def _result_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in results:
        rows.append(
            [
                result["case_id"],
                result["input_kind"],
                result["status"],
                f"{result['elapsed_sec']:.2f}",
                "x".join(str(value) for value in result["shape"]),
                result["output_path"],
                result["message"],
            ]
        )
    return rows


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _metrics_rows(metrics: list[Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in metrics:
        rows.append(
            [
                row.case_id,
                _format_metric(row.dice),
                _format_metric(row.iou),
                _format_metric(row.precision),
                _format_metric(row.recall),
                str(row.foreground),
                f"{row.foreground_ratio:.4%}",
                str(row.total),
                str(row.reference_path or ""),
                row.message,
            ]
        )
    return rows


def _metrics_summary(metrics: list[Any]) -> str:
    evaluated = [row for row in metrics if row.dice is not None]
    if not metrics:
        return "未生成指标。"
    if not evaluated:
        return "未提供匹配 reference mask，仅显示预测前景统计。"
    mean_dice = sum(row.dice for row in evaluated if row.dice is not None) / len(evaluated)
    mean_iou = sum(row.iou for row in evaluated if row.iou is not None) / len(evaluated)
    return (
        f"已评估 {len(evaluated)}/{len(metrics)} 个样本，"
        f"mean Dice={mean_dice:.4f}, mean IoU={mean_iou:.4f}"
    )


def _result_choices(results: list[dict[str, Any]]) -> list[str]:
    return [result["case_id"] for result in results if result["status"] == "done"]


def _selected_result(results: list[dict[str, Any]], case_id: str | None) -> dict[str, Any] | None:
    for result in results:
        if result["case_id"] == case_id:
            return result
    return None


def _get_cancel_event(cancel_token: str | None) -> tuple[str, threading.Event]:
    token = cancel_token or uuid.uuid4().hex
    with WEBUI_CANCEL_LOCK:
        event = WEBUI_CANCEL_EVENTS.setdefault(token, threading.Event())
    return token, event


def _remove_cancel_event(cancel_token: str) -> None:
    with WEBUI_CANCEL_LOCK:
        WEBUI_CANCEL_EVENTS.pop(cancel_token, None)


def run_inference_ui(
    model_key: str,
    checkpoint_dropdown: str,
    checkpoint_path: str,
    local_input_path: str,
    uploaded_files: list[object] | None,
    uploaded_directory: list[object] | None,
    reference_path: str,
    reference_files: list[object] | None,
    reference_directory: list[object] | None,
    output_dir: str,
    cancel_token: str | None,
    progress: gr.Progress = WEBUI_PROGRESS,
) -> Iterator[
    tuple[
        list[list[Any]],
        list[list[Any]],
        str,
        dict[str, Any],
        list[dict[str, Any]],
        None,
        str,
    ]
]:
    cancel_token, cancel_event = _get_cancel_event(cancel_token)
    cancel_event.clear()
    checkpoint = checkpoint_path.strip() or checkpoint_dropdown
    uploads = [*(uploaded_files or []), *(uploaded_directory or [])]
    yield (
        [],
        [],
        "正在收集输入文件...",
        gr.update(choices=[], value=None),
        [],
        None,
        cancel_token,
    )
    try:
        progress(0.05, desc="Collecting inputs")
        inputs = collect_inputs(local_input_path.strip() or None, uploads)
        input_summary = ", ".join(f"{item.case_id}({item.kind})" for item in inputs)
        yield (
            [],
            [],
            f"已加入 {len(inputs)} 个样本：{input_summary}\n正在启动 nnU-Net 推理，"
            "首次加载模型通常需要 1-3 分钟...",
            gr.update(choices=[], value=None),
            [],
            None,
            cancel_token,
        )
        progress(0.25, desc="Running nnU-Net prediction")
        results = run_batch_inference(
            model_key=model_key,
            inputs=inputs,
            checkpoint_path_or_name=checkpoint or None,
            output_dir=output_dir.strip() or DEFAULT_OUTPUT_DIR,
            config_path=DEFAULT_CONFIG,
            cancel_event=cancel_event,
        )
    except InferenceCancelledError as exc:
        _remove_cancel_event(cancel_token)
        yield (
            [],
            [],
            str(exc),
            gr.update(choices=[], value=None),
            [],
            None,
            cancel_token,
        )
        return
    except Exception as exc:
        _remove_cancel_event(cancel_token)
        yield (
            [],
            [],
            f"推理失败：{exc}",
            gr.update(choices=[], value=None),
            [],
            None,
            cancel_token,
        )
        return
    progress(0.95, desc="Preparing results")
    result_dicts = [
        {
            "case_id": result.case_id,
            "input_kind": result.input_kind,
            "input_path": str(result.input_path),
            "output_path": str(result.output_path),
            "elapsed_sec": result.elapsed_sec,
            "shape": result.shape,
            "status": result.status,
            "message": result.message,
        }
        for result in results
    ]
    try:
        reference_uploads = [*(reference_files or []), *(reference_directory or [])]
        references = collect_reference_masks(reference_path.strip() or None, reference_uploads)
        metrics = evaluate_result_metrics(results, references)
        metrics_summary = _metrics_summary(metrics)
    except Exception as exc:
        metrics = []
        metrics_summary = f"指标计算失败：{exc}"
    done = sum(1 for result in result_dicts if result["status"] == "done")
    failed = len(result_dicts) - done
    resolved_output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    log = (
        f"完成：{done}，失败：{failed}\n"
        f"{metrics_summary}\n"
        f"输出目录：{resolved_output_dir}"
    )
    choices = _result_choices(result_dicts)
    _remove_cancel_event(cancel_token)
    yield (
        _result_rows(result_dicts),
        _metrics_rows(metrics),
        log,
        gr.update(choices=choices, value=choices[0] if choices else None),
        result_dicts,
        None,
        cancel_token,
    )


def stop_inference_ui(cancel_token: str | None) -> str:
    if not cancel_token:
        return "没有正在运行的推理任务。"
    with WEBUI_CANCEL_LOCK:
        event = WEBUI_CANCEL_EVENTS.get(cancel_token)
    if event is None:
        return "没有正在运行的推理任务。"
    event.set()
    return "已请求终止推理，正在停止 nnU-Net 进程..."


def open_output_dir_ui(output_dir: str) -> str:
    path = Path(output_dir.strip() or DEFAULT_OUTPUT_DIR).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return f"已打开输出目录：{path}"


def clear_output_dir_ui(output_dir: str, confirm_clear: bool) -> str:
    if not confirm_clear:
        return "请先勾选“确认清理输出”。"
    path = Path(output_dir.strip() or DEFAULT_OUTPUT_DIR).resolve()
    project_root = PROJECT_ROOT.resolve()
    protected_paths = {
        project_root,
        project_root / "src",
        project_root / "tests",
        project_root / "configs",
        project_root / "docs",
        project_root / "dataset",
        project_root / "checkpoints",
        project_root / ".git",
    }
    try:
        path.relative_to(project_root)
    except ValueError:
        return f"拒绝清理项目外目录：{path}"
    if path in protected_paths or path.anchor == str(path):
        return f"拒绝清理受保护目录：{path}"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return f"输出目录不存在，已创建空目录：{path}"
    if not path.is_dir():
        return f"输出路径不是文件夹，未清理：{path}"

    removed = 0
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return f"已清理输出目录：{path}，删除 {removed} 个项目。"


def preview_result_ui(
    case_id: str | None,
    results: list[dict[str, Any]] | None,
    output_dir: str,
    preview_axis: str,
    slice_index: float | None,
) -> tuple[str | None, str]:
    if not case_id or not results:
        return None, ""
    result = _selected_result(results, case_id)
    if result is None:
        return None, ""
    preview_dir = Path(output_dir.strip() or DEFAULT_OUTPUT_DIR) / "_previews"
    requested_slice = None if slice_index is None else int(slice_index)
    preview = make_preview(
        result["input_path"],
        result["output_path"],
        preview_dir,
        axis=int(preview_axis),
        slice_index=requested_slice,
    )
    return str(preview), f"{result['case_id']} -> {result['output_path']}"


def refresh_checkpoints_ui() -> tuple[dict[str, Any], str]:
    choices = _checkpoint_choices()
    value = choices[0][1] if choices else ""
    return gr.update(choices=choices, value=value), value


def make_theme() -> gr.themes.ThemeClass:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as app:
        results_state = gr.State([])
        cancel_state = gr.State(None)
        with gr.Column(elem_id="imgseg-shell"):
            gr.HTML(
                """
                <div id="imgseg-header">
                  <h1>ImgSeg WebUI</h1>
                  <p>nnU-Net v2 batch inference for NIfTI and 2D images</p>
                </div>
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=360):
                    model = gr.Dropdown(
                        label="模型",
                        choices=_model_choices(),
                        value="nnunet_v2",
                        interactive=True,
                    )
                    checkpoint_dropdown = gr.Dropdown(
                        label="权重",
                        choices=_checkpoint_choices(),
                        value=_default_checkpoint(),
                        interactive=True,
                    )
                    checkpoint_path = gr.Textbox(
                        label="权重路径",
                        value=_default_checkpoint(),
                        placeholder="checkpoint_final.pth 或完整 .pth 路径",
                    )
                    refresh_checkpoints = gr.Button("刷新权重", variant="secondary")
                    local_input_path = gr.Textbox(
                        label="本机输入路径",
                        placeholder="单个 .nii/.nii.gz/.png/.jpg，或包含这些文件的文件夹",
                    )
                    uploaded_files = gr.File(
                        label="上传文件",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".png", ".jpg", ".jpeg"],
                    )
                    uploaded_directory = gr.File(
                        label="上传文件夹",
                        file_count="directory",
                        file_types=[".nii", ".nii.gz", ".png", ".jpg", ".jpeg"],
                    )
                    reference_path = gr.Textbox(
                        label="Reference mask 路径",
                        placeholder="可选：单个 mask 或包含 mask 的文件夹，用于计算 Dice/IoU",
                    )
                    reference_files = gr.File(
                        label="上传 reference mask",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".png", ".jpg", ".jpeg"],
                    )
                    reference_directory = gr.File(
                        label="上传 reference 文件夹",
                        file_count="directory",
                        file_types=[".nii", ".nii.gz", ".png", ".jpg", ".jpeg"],
                    )
                    with gr.Row():
                        output_dir = gr.Textbox(
                            label="输出目录",
                            value=str(DEFAULT_OUTPUT_DIR),
                            scale=3,
                        )
                        with gr.Column(scale=1, min_width=150):
                            open_output_button = gr.Button(
                                "打开输出文件夹", variant="secondary"
                            )
                            clear_output_button = gr.Button(
                                "清理输出文件夹", variant="secondary"
                            )
                    confirm_clear = gr.Checkbox(label="确认清理输出", value=False)
                    with gr.Row():
                        run_button = gr.Button(
                            "开始推理", variant="primary", elem_id="imgseg-run"
                        )
                        stop_button = gr.Button(
                            "终止推理", variant="secondary", elem_id="imgseg-stop"
                        )
                with gr.Column(scale=7, min_width=560):
                    status = gr.Textbox(
                        label="运行状态",
                        lines=3,
                        interactive=False,
                        elem_classes=["imgseg-status"],
                    )
                    table = gr.Dataframe(
                        headers=["case", "type", "status", "sec", "shape", "output", "message"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        label="结果",
                        interactive=False,
                        wrap=True,
                    )
                    metrics_table = gr.Dataframe(
                        headers=[
                            "case",
                            "Dice",
                            "IoU",
                            "Precision",
                            "Recall",
                            "foreground",
                            "foreground %",
                            "total",
                            "reference",
                            "message",
                        ],
                        datatype=[
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                        ],
                        label="指标 / 统计",
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        result_selector = gr.Dropdown(label="预览", choices=[], interactive=True)
                        preview_button = gr.Button(
                            "生成预览",
                            variant="secondary",
                            scale=1,
                            elem_id="imgseg-preview-button",
                        )
                    with gr.Row():
                        preview_axis = gr.Dropdown(
                            label="预览轴向",
                            choices=[
                                ("Axial / Z", "2"),
                                ("Coronal / Y", "1"),
                                ("Sagittal / X", "0"),
                            ],
                            value="2",
                            interactive=True,
                        )
                        preview_slice = gr.Number(
                            label="Slice index",
                            value=None,
                            precision=0,
                            minimum=0,
                            interactive=True,
                        )
                    preview_image = gr.Image(label="分割预览", type="filepath", height=560)
                    preview_status = gr.Textbox(label="预览状态", interactive=False)

            refresh_checkpoints.click(
                refresh_checkpoints_ui,
                outputs=[checkpoint_dropdown, checkpoint_path],
            )
            checkpoint_dropdown.change(lambda value: value, checkpoint_dropdown, checkpoint_path)
            run_event = run_button.click(
                run_inference_ui,
                inputs=[
                    model,
                    checkpoint_dropdown,
                    checkpoint_path,
                    local_input_path,
                    uploaded_files,
                    uploaded_directory,
                    reference_path,
                    reference_files,
                    reference_directory,
                    output_dir,
                    cancel_state,
                ],
                outputs=[
                    table,
                    metrics_table,
                    status,
                    result_selector,
                    results_state,
                    preview_image,
                    cancel_state,
                ],
            )
            stop_button.click(
                stop_inference_ui,
                inputs=cancel_state,
                outputs=status,
                queue=False,
                cancels=[run_event],
            )
            open_output_button.click(
                open_output_dir_ui,
                inputs=output_dir,
                outputs=status,
                queue=False,
            )
            clear_output_button.click(
                clear_output_dir_ui,
                inputs=[output_dir, confirm_clear],
                outputs=status,
                queue=False,
            )
            preview_button.click(
                preview_result_ui,
                inputs=[result_selector, results_state, output_dir, preview_axis, preview_slice],
                outputs=[preview_image, preview_status],
            )
            result_selector.change(
                preview_result_ui,
                inputs=[result_selector, results_state, output_dir, preview_axis, preview_slice],
                outputs=[preview_image, preview_status],
            )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--prevent-thread-lock", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_app().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=args.prevent_thread_lock,
        theme=make_theme(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
