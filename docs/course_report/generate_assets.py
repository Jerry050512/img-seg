"""Generate report figures from the available three-model experiment artifacts.

Run from the repository root with::

    uv run python docs/course_report/generate_assets.py

The script never modifies the source NIfTI files.  All generated figures are
written to ``docs/course_report/assets``.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import nibabel as nib
import numpy as np
import pandas as pd
import torch

plt.switch_backend("Agg")


available_fonts = {font.name for font in font_manager.fontManager.ttflist}
cjk_font = next(
    (
        name
        for name in (
            "LXGW WenKai",
            "Songti SC",
            "Source Han Serif",
            "SimSun",
            "Microsoft YaHei",
            "Noto Serif CJK SC",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "DejaVu Sans",
        )
        if name in available_fonts
    ),
    "DejaVu Sans",
)
plt.rcParams["font.family"] = [cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_RUN_ID = "nnunet_v2_2d_fold0_200epochs_20260710_114102"
RUN_ID = DEFAULT_RUN_ID
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
PRED_DIR = ROOT / "outputs" / f"{RUN_ID}_best"
TEST_DIR = ROOT / "dataset" / "processed" / "nnunet_raw" / "Dataset501_ImgSeg"
SEGRESNET_HISTORY = ROOT / "outputs" / "monai_segresnet" / "history.json"
SEGRESNET_PRED_DIR = ROOT / "outputs" / "monai_segresnet" / "test_predictions"
SEGRESNET_TEST_DIR = ROOT / "dataset"
COMPARISON_PATH = ASSETS / "model_comparison.json"
PLANS_PATH = (
    ROOT
    / "checkpoints"
    / "nnunet_v2_runs"
    / RUN_ID
    / "Dataset501_ImgSeg"
    / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
    / "plans.json"
)


def configure_nnunet_run(run_id: str) -> None:
    """Point report generation at a specific local nnU-Net run."""

    global RUN_ID, REPORT_DIR, PRED_DIR, PLANS_PATH
    RUN_ID = run_id
    REPORT_DIR = ROOT / "outputs" / "reports" / run_id
    PRED_DIR = ROOT / "outputs" / f"{run_id}_best"
    PLANS_PATH = (
        ROOT
        / "checkpoints"
        / "nnunet_v2_runs"
        / run_id
        / "Dataset501_ImgSeg"
        / "nnUNetTrainer_200epochs__nnUNetPlans__2d"
        / "plans.json"
    )


def require_generation_inputs() -> None:
    required = [
        REPORT_DIR / "epoch_metrics.csv",
        PRED_DIR / "metrics.json",
        PLANS_PATH,
        SEGRESNET_HISTORY,
        COMPARISON_PATH,
    ]
    for case_id in ("S_3", "2_25_XY"):
        required.extend(
            [
                TEST_DIR / "imagesTs" / f"{case_id}_0000.nii.gz",
                TEST_DIR / "labelsTs" / f"{case_id}.nii.gz",
                PRED_DIR / f"{case_id}.nii.gz",
                SEGRESNET_TEST_DIR / case_id / "image.nii.gz",
                SEGRESNET_TEST_DIR / case_id / "mask.nii.gz",
                SEGRESNET_PRED_DIR / f"{case_id}.nii.gz",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing report inputs for nnU-Net run {RUN_ID!r}:\n{formatted}\n"
            "Pass the correct run with `--run-id`, or restore the ignored local artifacts."
        )


COLORS = {
    "navy": "#17324d",
    "blue": "#2f6f8f",
    "teal": "#2a9d8f",
    "gold": "#e9c46a",
    "orange": "#f4a261",
    "red": "#e76f51",
    "magenta": "#c44569",
    "gray": "#667085",
}


def _load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def _slice_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    ref = reference.astype(bool)
    pred = prediction.astype(bool)
    axes = (0, 1)
    tp = np.logical_and(ref, pred).sum(axis=axes)
    fp = np.logical_and(~ref, pred).sum(axis=axes)
    fn = np.logical_and(ref, ~pred).sum(axis=axes)
    tn = np.logical_and(~ref, ~pred).sum(axis=axes)
    denom = 2 * tp + fp + fn
    dice = np.divide(2 * tp, denom, out=np.ones_like(tp, dtype=float), where=denom != 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "dice": dice}


def _select_slices(
    case_id: str, reference: np.ndarray, prediction: np.ndarray
) -> list[tuple[str, int]]:
    stats = _slice_metrics(reference, prediction)
    ref_area = reference.astype(bool).sum(axis=(0, 1))
    valid = np.flatnonzero(ref_area >= np.percentile(ref_area[ref_area > 0], 60))
    if case_id == "S_3":
        best = int(valid[np.argmax(stats["dice"][valid])])
        return [("边界清晰且高度吻合", best)]

    fn_order = valid[np.argsort(stats["fn"][valid])[::-1]]
    fn_slice = int(fn_order[0])
    fp_order = valid[np.argsort(stats["fp"][valid])[::-1]]
    fp_slice = next(
        (int(idx) for idx in fp_order if abs(int(idx) - fn_slice) > 5),
        int(fp_order[0]),
    )
    return [("弱边界导致漏分", fn_slice), ("复杂边界导致误分", fp_slice)]


def _draw_bad_case(
    case_id: str,
    category: str,
    index: int,
    image: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    output_name: str,
) -> None:
    img = np.rot90(_normalize(image[:, :, index]))
    ref = np.rot90(reference[:, :, index].astype(bool))
    pred = np.rot90(prediction[:, :, index].astype(bool))
    tp = ref & pred
    fp = ~ref & pred
    fn = ref & ~pred
    tp_n, fp_n, fn_n = int(tp.sum()), int(fp.sum()), int(fn.sum())
    dice = 2 * tp_n / max(2 * tp_n + fp_n + fn_n, 1)

    error_rgba = np.zeros((*ref.shape, 4), dtype=float)
    error_rgba[tp] = (42 / 255, 157 / 255, 143 / 255, 0.58)
    error_rgba[fp] = (244 / 255, 162 / 255, 97 / 255, 0.78)
    error_rgba[fn] = (196 / 255, 69 / 255, 105 / 255, 0.82)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), dpi=180)
    for ax in axes:
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    axes[0].set_title("原始切片", fontsize=11, color=COLORS["navy"])
    if ref.any():
        axes[1].contour(ref, levels=[0.5], colors=["#35d07f"], linewidths=1.1)
    if pred.any():
        axes[1].contour(pred, levels=[0.5], colors=["#ff4d6d"], linewidths=1.0)
    axes[1].set_title("轮廓：绿=标注，红=预测", fontsize=11, color=COLORS["navy"])
    axes[2].imshow(error_rgba)
    axes[2].set_title("误差：青=TP，橙=FP，紫=FN", fontsize=11, color=COLORS["navy"])
    fig.suptitle(
        f"{category} | {case_id} | z={index} | slice Dice={dice:.4f}",
        fontsize=13,
        fontweight="bold",
        color=COLORS["navy"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(ASSETS / output_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_bad_cases() -> None:
    outputs: list[tuple[str, str, int, str]] = []
    for case_id in ("S_3", "2_25_XY"):
        image = _load(TEST_DIR / "imagesTs" / f"{case_id}_0000.nii.gz")
        reference = _load(TEST_DIR / "labelsTs" / f"{case_id}.nii.gz") > 0
        prediction = _load(PRED_DIR / f"{case_id}.nii.gz") > 0
        selections = _select_slices(case_id, reference, prediction)
        for category, index in selections:
            output_name = f"bad_case_{len(outputs) + 1}.png"
            _draw_bad_case(
                case_id,
                category,
                index,
                image,
                reference,
                prediction,
                output_name,
            )
            outputs.append((case_id, category, index, output_name))
        del image, reference, prediction

    with (ASSETS / "bad_case_slices.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"case_id": case_id, "category": category, "slice": index, "asset": asset}
                for case_id, category, index, asset in outputs
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def generate_segresnet_bad_cases() -> None:
    outputs: list[tuple[str, str, int, str]] = []
    for case_id in ("S_3", "2_25_XY"):
        image = _load(SEGRESNET_TEST_DIR / case_id / "image.nii.gz")
        reference = _load(SEGRESNET_TEST_DIR / case_id / "mask.nii.gz") > 0
        prediction = _load(SEGRESNET_PRED_DIR / f"{case_id}.nii.gz") > 0
        for category, index in _select_slices(case_id, reference, prediction):
            output_name = f"segresnet_case_{len(outputs) + 1}.png"
            _draw_bad_case(
                case_id,
                category,
                index,
                image,
                reference,
                prediction,
                output_name,
            )
            outputs.append((case_id, category, index, output_name))
        del image, reference, prediction

    with (ASSETS / "segresnet_case_slices.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"case_id": case_id, "category": category, "slice": index, "asset": asset}
                for case_id, category, index, asset in outputs
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def generate_training_curves() -> None:
    frame = pd.read_csv(REPORT_DIR / "epoch_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), dpi=180)
    axes[0].plot(frame["epoch"], frame["train_loss"], color=COLORS["blue"], lw=1.4, label="Train")
    axes[0].plot(
        frame["epoch"],
        frame["val_loss"],
        color=COLORS["red"],
        lw=1.25,
        label="Validation",
    )
    axes[0].axvline(120, color=COLORS["gold"], ls="--", lw=1, label="Best epoch 120")
    axes[0].set(title="Loss convergence", xlabel="Epoch", ylabel="Dice + CE loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(frame["epoch"], frame["pseudo_dice"], color=COLORS["teal"], lw=1.4)
    axes[1].scatter([120], [0.8375], color=COLORS["red"], s=30, zorder=3)
    axes[1].annotate("best 0.8375", (120, 0.8375), xytext=(128, 0.845), fontsize=8)
    axes[1].set(title="Validation pseudo Dice", xlabel="Epoch", ylabel="Dice", ylim=(0.64, 0.86))

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_segresnet_training_curve() -> None:
    history = json.loads(SEGRESNET_HISTORY.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    validation = [row for row in history if "validation" in row]
    val_epochs = [row["epoch"] for row in validation]
    val_dice = [row["validation"]["summary"]["dice"] for row in validation]
    best_index = int(np.argmax(val_dice))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), dpi=180)
    axes[0].plot(epochs, losses, color=COLORS["blue"], lw=1.15)
    smoothed = pd.Series(losses).rolling(10, min_periods=1, center=True).mean()
    axes[0].plot(epochs, smoothed, color=COLORS["gold"], lw=2, label="10-epoch mean")
    axes[0].set(title="SegResNet training loss", xlabel="Epoch", ylabel="Dice + BCE loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(val_epochs, val_dice, color=COLORS["teal"], marker="o", ms=3, lw=1.4)
    axes[1].scatter(
        [val_epochs[best_index]], [val_dice[best_index]], color=COLORS["red"], s=35, zorder=3
    )
    axes[1].annotate(
        f"best {val_dice[best_index]:.4f}",
        (val_epochs[best_index], val_dice[best_index]),
        xytext=(-42, 15),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set(
        title="SegResNet validation Dice",
        xlabel="Epoch",
        ylabel="Dice",
        ylim=(0.55, 0.86),
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "segresnet_training_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_radar() -> None:
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    labels = ["mIoU", "Dice", "Accuracy", "Precision", "Recall"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 5.6), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    styles = [
        ("nnunet_v2", COLORS["blue"]),
        ("monai_segresnet", COLORS["teal"]),
        ("efficientnet_b0", COLORS["orange"]),
    ]
    for key, color in styles:
        row = comparison[key]
        values = [row["miou"], row["dice"], row["accuracy"], row["precision"], row["recall"]]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.8, label=row["label"])
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels([".80", ".85", ".90", ".95", "1.0"], fontsize=7, color=COLORS["gray"])
    ax.grid(alpha=0.25)
    ax.set_title("Three-model test profile", pad=20, color=COLORS["navy"], fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.04), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "metric_radar.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def measure_network() -> None:
    """Measure nnU-Net 2.x using APIs validated against project version 2.5.x."""

    try:
        nnunet_version = version("nnunetv2")
    except PackageNotFoundError as exc:
        raise RuntimeError("nnunetv2 is required to measure the nnU-Net network") from exc
    if nnunet_version.split(".", 1)[0] != "2":
        raise RuntimeError(
            f"Unsupported nnunetv2 {nnunet_version}; internal architecture APIs "
            "were validated with nnunetv2 2.5.x."
        )
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = json.loads(PLANS_PATH.read_text(encoding="utf-8"))
    plans_manager = PlansManager(plans)
    configuration = plans_manager.get_configuration("2d")
    model = nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration,
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=False,
    ).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    flops = 0

    def count_conv(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal flops
        kernel = int(np.prod(module.kernel_size))
        if isinstance(module, torch.nn.ConvTranspose2d):
            multiply_adds = (
                int(inputs[0].numel()) * kernel * module.out_channels // module.groups
            )
        else:
            multiply_adds = int(output.numel()) * kernel * module.in_channels // module.groups
        bias_adds = int(output.numel()) if module.bias is not None else 0
        flops += 2 * multiply_adds + bias_adds

    handles = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(count_conv))
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    with torch.inference_mode():
        model(torch.zeros(1, 1, 512, 512))
    for handle in handles:
        handle.remove()

    payload = {
        "input": [1, 1, 512, 512],
        "parameters": params,
        "parameters_million": params / 1e6,
        "conv_flops": flops,
        "conv_flops_giga": flops / 1e9,
        "counting_convention": (
            "one multiply and one add are counted as two FLOPs; "
            "convolution and transposed convolution only"
        ),
    }
    with (ASSETS / "network_complexity.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def measure_segresnet_network() -> None:
    from monai.networks.nets import SegResNet

    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
        norm="INSTANCE",
    ).eval()
    flops = 0

    def count_conv(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal flops
        kernel = int(np.prod(module.kernel_size))
        if isinstance(module, torch.nn.ConvTranspose3d):
            multiply_adds = int(inputs[0].numel()) * kernel * module.out_channels // module.groups
        else:
            multiply_adds = int(output.numel()) * kernel * module.in_channels // module.groups
        flops += 2 * multiply_adds + (int(output.numel()) if module.bias is not None else 0)

    handles = [
        module.register_forward_hook(count_conv)
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d))
    ]
    with torch.inference_mode():
        model(torch.zeros(1, 1, 96, 96, 64))
    for handle in handles:
        handle.remove()
    params = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "input": [1, 1, 96, 96, 64],
        "parameters": params,
        "parameters_million": params / 1e6,
        "conv_flops": flops,
        "conv_flops_giga": flops / 1e9,
        "counting_convention": (
            "one multiply and one add are counted as two FLOPs; "
            "3D convolution and transposed convolution only"
        ),
    }
    with (ASSETS / "segresnet_complexity.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Local nnU-Net run id used for metrics, predictions, and plans.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_nnunet_run(args.run_id)
    require_generation_inputs()
    ASSETS.mkdir(parents=True, exist_ok=True)
    generate_training_curves()
    generate_segresnet_training_curve()
    generate_radar()
    generate_bad_cases()
    generate_segresnet_bad_cases()
    measure_network()
    measure_segresnet_network()
    print(f"Generated report assets in {ASSETS}")


if __name__ == "__main__":
    main()
