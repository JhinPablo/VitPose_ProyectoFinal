from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate_paths(path: Path) -> Iterable[Path]:
	yield path
	if not path.is_absolute():
		yield PROJECT_ROOT / path


def validate_path(path_str: str, must_exist: bool = True) -> Path:
	path = Path(path_str).expanduser()
	for candidate in _candidate_paths(path):
		candidate = candidate.resolve(strict=False)
		if not must_exist or candidate.exists():
			return candidate
	raise FileNotFoundError(f"No se encontró la ruta: {path_str} (intenta rutas relativas a {PROJECT_ROOT})")
