"""Build the lean Datathon ZIP and check its required contents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "final_package"
ARCHIVE = ROOT / "submission_ready" / "DevOps_Datathon.zip"

# The JSON reports listed here are read directly by the submitted notebook.
FILES = (
    "DevOps_FinalNotebook.ipynb",
    "README.md",
    "requirements.txt",
    "submission_task1.csv",
    "submission_task2a.csv",
    "submission_task2b.csv",
    "models/task1_models.pkl",
    "models/task2a_models.pkl",
    "reports/architecture_diagrams.pdf",
    "reports/data_preprocessing.pdf",
    "reports/task2b_prioritization_policy.pdf",
    "reports/ai_tool_disclosure.pdf",
    "reports/error_analysis.json",
    "reports/final_audit.json",
    "reports/initial_audit.json",
    "reports/task1_gpu_benchmark.json",
    "reports/task1_label_audit.json",
    "reports/task2a_model_report.json",
    "reports/task2b_optimization_report.json",
    "src/audit.py",
    "src/audit_data.py",
    "src/evaluate.py",
    "src/features_task1.py",
    "src/features_task2a.py",
    "src/inference.py",
    "src/labels.py",
    "src/optimize_task2b.py",
    "src/project_paths.py",
    "src/train_task1.py",
    "src/train_task1_gpu.py",
    "src/train_task2a.py",
)


def main() -> None:
    missing = [name for name in FILES if not (PACKAGE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package files: {missing}")

    notebook = json.loads((PACKAGE / "DevOps_FinalNotebook.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    notebook_reports = set(re.findall(r'"([a-z0-9_]+\.json)"', source))
    # Only report filenames referenced under WORK / "reports" are included.
    missing_reports = {f"reports/{name}" for name in notebook_reports} - set(FILES)
    if missing_reports:
        raise AssertionError(f"Notebook report dependencies missing from ZIP: {sorted(missing_reports)}")

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in FILES:
            archive.write(PACKAGE / name, name)

    with ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise AssertionError("ZIP integrity check failed")
        if set(archive.namelist()) != set(FILES):
            raise AssertionError("ZIP member list differs from the approved file list")
        for name in FILES:
            if archive.read(name) != (PACKAGE / name).read_bytes():
                raise AssertionError(f"ZIP member differs from source: {name}")

    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"WROTE {ARCHIVE} ({len(FILES)} files, {ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256 {digest}")
    print("ZIP integrity and notebook report dependencies: PASSED")


if __name__ == "__main__":
    main()
