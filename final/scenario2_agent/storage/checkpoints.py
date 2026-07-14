"""Guardado y carga de checkpoints, con toda la información para reanudar o para ver jugar.

Un checkpoint guarda los pesos del modelo, el estado del optimizer (para reanudar el entrenamiento
sin perder el momentum de Adam) y un diccionario `meta` con la configuración necesaria para
reconstruir el modelo y el codificador de observación (para que la herramienta visual pueda cargar
el checkpoint sin conocer los hiperparámetros de antemano).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scenario2_agent.models.mlp_policy import MLPPolicyValue


def save_checkpoint(path: str | Path, model: MLPPolicyValue, optimizer: torch.optim.Optimizer | None, meta: dict[str, Any]) -> None:
    """Guarda pesos del modelo, estado del optimizer y metadatos en `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "meta": meta,
    }
    torch.save(payload, path)


def build_model_from_meta(meta: dict[str, Any]) -> MLPPolicyValue:
    """Reconstruye un `MLPPolicyValue` a partir de la configuración guardada en `meta['model']`."""
    model_config = meta["model"]
    return MLPPolicyValue(
        input_dim=int(model_config["input_dim"]),
        num_actions=int(model_config["num_actions"]),
        hidden_sizes=tuple(model_config["hidden_sizes"]),
        activation=str(model_config["activation"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[MLPPolicyValue, dict[str, Any]]:
    """Carga un checkpoint y devuelve (modelo reconstruido y en modo eval, payload completo).

    El payload incluye `optimizer_state` y `meta`, útiles para reanudar el entrenamiento o para
    reconstruir el codificador de observación en la herramienta visual.
    """
    path = Path(path)
    # weights_only=False porque el payload incluye `meta` (un dict con configuración, no solo tensores).
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model_from_meta(payload["meta"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
