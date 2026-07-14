"""Mapeo entre las 5 acciones del modelo y las acciones de Overcooked.

El modelo produce un índice en [0, 5): 0 arriba, 1 abajo, 2 derecha, 3 izquierda, 4 interactuar.
Deliberadamente NO existe "quedarse quieto" (STAY), para forzar que el agente no se quede parado.
Este módulo traduce ese índice al objeto de acción que espera `OvercookedEnv.step`.
"""

from __future__ import annotations

from overcooked_ai_py.mdp.actions import Action, Direction

# Orden fijo de las acciones del agente. El índice de esta lista es la salida del modelo.
AGENT_ACTIONS = [
    Direction.NORTH,  # 0 arriba
    Direction.SOUTH,  # 1 abajo
    Direction.EAST,   # 2 derecha
    Direction.WEST,   # 3 izquierda
    Action.INTERACT,  # 4 interactuar
]

NUM_AGENT_ACTIONS = len(AGENT_ACTIONS)

# Índice de la acción "interactuar" dentro de AGENT_ACTIONS (usado por los detectores de eventos).
INTERACT_ACTION_INDEX = 4


def agent_action_to_overcooked(action_index: int):
    """Convierte un índice de acción del modelo (0..4) en la acción de Overcooked correspondiente."""
    if not 0 <= int(action_index) < NUM_AGENT_ACTIONS:
        raise ValueError(f"Índice de acción fuera de rango [0, {NUM_AGENT_ACTIONS}): {action_index}")
    return AGENT_ACTIONS[int(action_index)]
