from __future__ import annotations

from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"


def _default(script: str, variable: str) -> str:
    text = (SLURM / script).read_text(encoding="utf-8")
    match = re.search(
        rf'^{variable}="\$\{{{variable}:-([^}}]+)\}}"$',
        text,
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_msi_shell_scripts_are_valid_bash() -> None:
    scripts = (
        SLURM / "setup_env_msi.sh",
        SLURM / "generate_data_msi.sbatch",
        SLURM / "train_msi.sbatch",
    )
    subprocess.run(
        ["bash", "-n", *(str(path) for path in scripts)],
        check=True,
    )


def test_msi_generator_and_trainer_share_production_defaults() -> None:
    assert _default(
        "generate_data_msi.sbatch",
        "CONFIG",
    ) == _default("train_msi.sbatch", "CONFIG")
    assert _default(
        "generate_data_msi.sbatch",
        "OUTPUT_DIR",
    ).replace("OUTPUT_DIR", "DATA_DIR") == _default(
        "train_msi.sbatch",
        "DATA_DIR",
    )


def test_msi_setup_uses_official_pytorch_wheel() -> None:
    text = (SLURM / "setup_env_msi.sh").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cu126" in text
    assert "conda install pytorch" not in text
    assert "conda create --copy" in text


def test_refined_config_differs_from_pilot_only_by_mesh_and_study_id() -> None:
    pilot = yaml.safe_load(
        (ROOT / "conf/nonlinear-pilot.yaml").read_text(encoding="utf-8")
    )
    refined = yaml.safe_load(
        (
            ROOT / "conf/nonlinear-production-36x48.yaml"
        ).read_text(encoding="utf-8")
    )
    pilot["study_id"] = refined["study_id"]
    pilot["mesh"] = refined["mesh"]
    assert pilot == refined
