"""Mapeamento unico das cameras do projeto para as variaveis do .env.

Centralizado aqui para que adicionar uma camera nova exija editar um lugar so.
"""
import os

CAMERA_ENV = {
    "cam26": "CAMERA_26_URL",
    "cam27": "CAMERA_27_URL",
    "cabine": "RTSP_URL",
    "cabine2": "CAMERA_46_URL",
}

# Cameras que enxergam a MESMA fila de ganchos por angulos diferentes. Cada
# grupo pode ser fundido: um gancho ambiguo em uma vista costuma estar claro na
# outra. Exige que as calibracoes usem a mesma numeracao fisica de gancho.
VISTAS_DA_CABINE = ("cabine", "cabine2")

CAMERA_NAMES = sorted(CAMERA_ENV)

# Nome curto para caber no painel. Bate com o jeito que a fabrica se refere a
# elas (o final do IP), que e mais util na tela do que "cabine2".
CAMERA_LABEL = {
    "cabine": "45",
    "cabine2": "46",
    "cam26": "26",
    "cam27": "27",
}


def label(name: str) -> str:
    return CAMERA_LABEL.get(name, name)


def vista_irma(name: str) -> str | None:
    """A outra camera que enxerga a mesma fila de ganchos, se houver."""
    if name not in VISTAS_DA_CABINE:
        return None
    outras = [c for c in VISTAS_DA_CABINE if c != name]
    return outras[0] if outras else None


def camera_url(name: str) -> str:
    """URL RTSP da camera, ou erro claro se nao estiver configurada no .env."""
    env_key = CAMERA_ENV[name]
    url = os.environ.get(env_key)
    if not url:
        raise SystemExit(f"Defina {env_key} no .env para usar a camera '{name}'")
    return url


def configured_cameras() -> dict[str, str]:
    """Todas as cameras que tem URL definida no .env."""
    return {name: os.environ[key] for name, key in CAMERA_ENV.items()
            if os.environ.get(key)}
