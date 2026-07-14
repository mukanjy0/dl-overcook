"""Codificador GUIADO (arquitectura jerárquica): parche local + recomendación de a dónde ir.

A diferencia del `EgocentricEncoder` (que da un vector a CADA tipo de objeto y deja que el agente
aprenda cuál importa), aquí un `target_provider` calcula por detrás el objetivo actual (la "receta")
y la observación solo trae:

- PARCHE local one-hot (contexto inmediato: paredes y qué tiles hay al lado, para interactuar).
- DIRECCIÓN del siguiente paso del camino más corto (BFS) hacia el objetivo actual.
- DISTANCIA (normalizada) al objetivo actual.
- FLAG de presencia de objetivo.
- OBJETO EN MANO (one-hot).

NO se incluye la orientación del jugador a propósito: que aprenda solo a mirar el objetivo antes de
interactuar (moverse contra el tile lo orienta). El MLP solo tiene que aprender a seguir la dirección
recomendada y a interactuar cuando la distancia es pequeña, lo cual es muy fácil y entrena rápido.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from scenario2_agent.environment import pathfinding
from scenario2_agent.observations.egocentric_encoder import (
    HELD_OBJECT_CATEGORIES,
    NUM_MAP_CHANNELS,
    build_local_patch,
)

# Vecinos de 4-conectividad (para medir la distancia a una celda contigua al objetivo).
_STEPS = [(0, -1), (0, 1), (1, 0), (-1, 0)]

# Número de valores del bloque de objetivo: dirección (dx, dy) + distancia + presencia.
_TARGET_VALUES = 4

# Valores del bloque de estado de la olla más cercana: [cebollas/máx, llena, cocinando].
_POT_STATE_VALUES = 3
# Máximo de ingredientes de una olla (receta de 3 cebollas).
_MAX_POT_ONIONS = 3


class GuidedEncoder:
    """Codifica el estado como parche local + recomendación (dirección/distancia) al objetivo actual."""

    observation_type = "guided"

    def __init__(self, target_provider: Callable, patch_radius: int = 1):
        """Inicializa el codificador guiado.

        `target_provider(state, mdp, agent_index)` devuelve las celdas objetivo del momento (la receta
        de alto nivel). `patch_radius` es el radio del parche local (lado = 2*r+1).
        """
        self.target_provider = target_provider
        self.patch_radius = int(patch_radius)
        self.patch_side = 2 * self.patch_radius + 1

    @property
    def observation_dim(self) -> int:
        """Devuelve la longitud fija del vector de observación."""
        patch_size = self.patch_side * self.patch_side * NUM_MAP_CHANNELS
        return patch_size + _TARGET_VALUES + _POT_STATE_VALUES + len(HELD_OBJECT_CATEGORIES)

    def encode(self, state, mdp, agent_index: int = 0) -> np.ndarray:
        """Codifica el estado en el vector de observación guiado del jugador `agent_index`."""
        player = state.players[agent_index]
        player_position = tuple(player.position)

        patch = build_local_patch(state, mdp, agent_index, player_position, self.patch_radius)
        target_block = self._encode_target(state, mdp, agent_index, player_position)
        pot_state = self._encode_pot_state(state, mdp, player_position)
        held = self._encode_held_object(player.held_object)

        return np.concatenate([patch.reshape(-1), target_block, pot_state, held]).astype(np.float32)

    def _encode_pot_state(self, state, mdp, player_position: tuple[int, int]) -> np.ndarray:
        """Devuelve [cebollas/máx, llena, cocinando] de la olla más cercana al jugador.

        Le da al agente la información que le faltaba para decidir CUÁNDO activar el horno: solo debe
        interactuar en vacío con la olla si está llena. Sin esto, "manos vacías pegado a la olla" se ve
        casi igual con 1 o con 3 cebollas, y cocina prematuramente. También sirve para P3+ (saber si la
        sopa ya está lista para sacarla con el plato).
        """
        block = np.zeros(_POT_STATE_VALUES, dtype=np.float32)
        pot_positions = mdp.get_pot_locations()
        if not pot_positions:
            return block
        player_x, player_y = player_position
        nearest = min(pot_positions, key=lambda p: abs(p[0] - player_x) + abs(p[1] - player_y))
        if state.has_object(nearest):
            soup = state.get_object(nearest)
            block[0] = min(len(soup.ingredients) / _MAX_POT_ONIONS, 1.0)
            block[1] = 1.0 if soup.is_full else 0.0
            block[2] = 1.0 if (soup.is_cooking or soup.is_ready) else 0.0
        return block

    def _encode_target(self, state, mdp, agent_index: int, player_position: tuple[int, int]) -> np.ndarray:
        """Devuelve [dir_x, dir_y, distancia_normalizada, presencia] hacia el objetivo actual.

        La dirección es el siguiente paso del camino más corto (BFS) rodeando paredes (y al otro
        jugador, tratado como celda bloqueada). La distancia son los pasos hasta ponerse contiguo al
        objetivo, normalizados por el tamaño del mapa. Si no hay objetivo o es inalcanzable, la
        presencia queda en 0 y la dirección en (0, 0).
        """
        block = np.zeros(_TARGET_VALUES, dtype=np.float32)
        target_cells = list(self.target_provider(state, mdp, agent_index))
        if not target_cells:
            return block

        other_player_positions = {
            tuple(other.position) for index, other in enumerate(state.players) if index != agent_index
        }
        walkable = pathfinding.walkable_cells(mdp)
        distances, parents = pathfinding.bfs_from(player_position, walkable, set(other_player_positions))

        block[3] = 1.0  # hay un objetivo definido
        direction = pathfinding.next_step_direction(player_position, target_cells, distances, parents)
        if direction is not None:
            block[0] = float(direction[0])
            block[1] = float(direction[1])

        terrain = mdp.terrain_mtx
        height = len(terrain)
        width = len(terrain[0]) if height > 0 else 0
        steps_to_target = _distance_to_target(target_cells, distances)
        if steps_to_target is not None:
            block[2] = steps_to_target / max(width + height, 1)
        else:
            block[2] = 1.0  # inalcanzable: se marca como "lejos"
        return block

    def _encode_held_object(self, held_object) -> np.ndarray:
        """Devuelve el one-hot del objeto en mano; índice 0 si no lleva nada."""
        vector = np.zeros(len(HELD_OBJECT_CATEGORIES), dtype=np.float32)
        if held_object is None:
            vector[0] = 1.0
        else:
            category_index = (
                HELD_OBJECT_CATEGORIES.index(held_object.name) if held_object.name in HELD_OBJECT_CATEGORIES else 0
            )
            vector[category_index] = 1.0
        return vector


def _distance_to_target(target_cells: list[tuple[int, int]], distances: dict) -> int | None:
    """Pasos hasta ponerse contiguo al objetivo más cercano alcanzable (None si ninguno lo es)."""
    best: int | None = None
    for target_x, target_y in target_cells:
        for delta_x, delta_y in _STEPS:
            neighbor = (target_x + delta_x, target_y + delta_y)
            if neighbor in distances:
                candidate = distances[neighbor]
                if best is None or candidate < best:
                    best = candidate
    return best
