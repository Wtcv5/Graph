from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


DEFAULT_OUTPUT_FILE = "geological_model.nc"
DEFAULT_INFO_FILE = "preprocessing_info.json"


@dataclass(frozen=True)
class DatasetConfig:
    project_root: Path
    config_path: Path
    name: str
    description: str
    data_dir: Path
    vp_file: str
    vs_file: str
    output_dir: Path
    grid_step_m: float
    chunk_size: int
    fill_kernel_size: int
    fill_max_iter: int
    title: str
    source: str
    coordinate_system: str
    density_formula: str

    @property
    def vp_path(self) -> Path:
        return self.data_dir / self.vp_file

    @property
    def vs_path(self) -> Path:
        return self.data_dir / self.vs_file

    @property
    def output_nc_path(self) -> Path:
        return self.output_dir / DEFAULT_OUTPUT_FILE

    @property
    def output_info_path(self) -> Path:
        return self.output_dir / DEFAULT_INFO_FILE

    def with_output_dir(self, output_dir: Path) -> "DatasetConfig":
        return replace(self, output_dir=output_dir)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_dataset_config(config_path: Path, project_root: Path | None = None) -> DatasetConfig:
    resolved_config_path = config_path.resolve()
    resolved_project_root = project_root.resolve() if project_root else resolved_config_path.parents[2]

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return DatasetConfig(
        project_root=resolved_project_root,
        config_path=resolved_config_path,
        name=raw["name"],
        description=raw["description"],
        data_dir=_resolve_project_path(resolved_project_root, raw["data_dir"]),
        vp_file=raw["vp_file"],
        vs_file=raw["vs_file"],
        output_dir=_resolve_project_path(resolved_project_root, raw["output_dir"]),
        grid_step_m=float(raw["grid_step_m"]),
        chunk_size=int(raw["chunk_size"]),
        fill_kernel_size=int(raw["fill_kernel_size"]),
        fill_max_iter=int(raw["fill_max_iter"]),
        title=raw["title"],
        source=raw["source"],
        coordinate_system=raw["coordinate_system"],
        density_formula=raw["density_formula"],
    )
