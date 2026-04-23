from pathlib import Path

from decouple import AutoConfig


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_ROOT = BASE_DIR.parent

config = AutoConfig(search_path=str(ENV_ROOT))


def config_bool(name, default=False):
    raw = config(name, default=None)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default
