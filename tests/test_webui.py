from __future__ import annotations

from pathlib import Path

from gradio.data_classes import FileData

from img_seg.inference.batch import collect_inputs
from img_seg.webui.app import UPLOAD_FILE_TYPES, build_app


def test_webui_accepts_generic_uploads_and_validates_compound_suffix(tmp_path: Path) -> None:
    nifti = tmp_path / "sample.nii.gz"
    nifti.write_bytes(b"upload path is validated before NIfTI loading")

    upload = FileData(path=str(nifti), orig_name="sample.nii.gz")
    inputs = collect_inputs(uploaded_files=[upload])

    assert UPLOAD_FILE_TYPES == ["file"]
    assert inputs[0].source_path == nifti.resolve()
    assert inputs[0].kind == "nifti"


def test_webui_builds_with_generic_file_upload_components() -> None:
    app = build_app()

    assert app is not None
