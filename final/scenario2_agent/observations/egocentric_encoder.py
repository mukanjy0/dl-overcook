"""Codificador de observación egocéntrico y semántico (híbrido) para el agente.

Todo es relativo al jugador (egocéntrico), combinando dos fuentes de información:

1. PARCHE LOCAL: un cuadrado pequeño centrado en el jugador con un canal one-hot por tipo de
   tile (pared, cebolla, horno, plato, salida). Sirve para lo cercano: esquivar paredes y saber
   si el jugador está adyacente/mirando algo para interactuar. Como los vectores relativos ya
   aportan la orientación de largo alcance, el parche puede ser pequeño (radio 1 = 3x3).

2. VECTORES RELATIVOS: para cada tipo de objeto (cebolla, horno, plato, salida, otro jugador) se
   da el vector `(dx, dy)` al más cercano, normalizado, más un flag de presencia. Esto resuelve
   la observabilidad parcial en mapas grandes (10x10+): el parche local no ve objetos lejanos,
   pero el vector sí indica hacia dónde está cada tipo de objeto.

Notas de diseño:
- "Otro jugador = pared": en el parche local, otro jugador se marca en el canal de pared, porque
  para navegar es simplemente un obstáculo (no se puede pasar por encima). Su posición para
  coordinar se conserva aparte, como vector relativo.
- El slot de `other_player` en los vectores relativos se mantiene aunque en Part 1 (un solo
  jugador) valga 0: así el tamaño del input NO cambia al pasar a la fase cooperativa y el
  warm-start del currículum sigue siendo válido.
- Se preserva la SEMÁNTICA: no se le dice al agente "ve aquí"; recibe vectores a TODOS los tipos
  y debe aprender por recompensa cuál importa en cada stage (en P1, la cebolla).

La observación es un vector plano de tamaño FIJO independiente del tamaño del nivel, para que el
mismo MLP sirva en todos los stages.
"""

from __future__ import annotations

import numpy as np
from overcooked_ai_py.mdp.actions import Direction

from scenario2_agent.environment import pathfinding

# Canales del parche local. Cada celda del parche activa (=1) el canal correspondiente.
# Otro jugador NO tiene canal propio: se marca como pared (obstáculo) en el parche.
TILE_CHANNELS: dict[str, int] = {
    "X": 0,  # pared / counter (y también otro jugador, tratado como obstáculo)
    "O": 1,  # dispensador de cebolla
    "P": 2,  # horno / olla
    "D": 3,  # dispensador de platos
    "S": 4,  # salida (serving)
}
WALL_CHANNEL = TILE_CHANNELS["X"]
NUM_MAP_CHANNELS = 5

# Tipos de objeto para los vectores relativos, en orden fijo. Cada uno aporta (dx, dy, presencia).
# `other_player` se conserva por compatibilidad con la fase cooperativa (vale 0 con un solo jugador).
OBJECT_TYPES: list[str] = ["onion", "pot", "dish", "serve", "other_player"]
VALUES_PER_OBJECT = 3

# Categorías del objeto en mano (one-hot). El índice 0 = "no lleva nada".
# Sin `tomato`: la receta es de 3 cebollas, así que nunca aparece un tomate en la mano.
HELD_OBJECT_CATEGORIES: list[str] = ["none", "onion", "dish", "soup"]

NUM_ORIENTATIONS = 4  # NORTH, SOUTH, EAST, WEST


def build_local_patch(state, mdp, agent_index: int, player_position: tuple[int, int], patch_radius: int) -> np.ndarray:
    """Construye el parche local one-hot centrado en el jugador (compartido por los codificadores).

    Cada celda activa el canal de su tile (pared/O/P/D/S). Fuera del grid y otro jugador se marcan
    como pared (obstáculos para navegar). Devuelve un array (lado, lado, NUM_MAP_CHANNELS).
    """
    patch_side = 2 * patch_radius + 1
    player_x, player_y = player_position
    other_player_positions = {
        tuple(other.position) for index, other in enumerate(state.players) if index != agent_index
    }
    terrain = mdp.terrain_mtx
    height = len(terrain)
    width = len(terrain[0]) if height > 0 else 0

    patch = np.zeros((patch_side, patch_side, NUM_MAP_CHANNELS), dtype=np.float32)
    for patch_row, delta_y in enumerate(range(-patch_radius, patch_radius + 1)):
        for patch_col, delta_x in enumerate(range(-patch_radius, patch_radius + 1)):
            cell_x = player_x + delta_x
            cell_y = player_y + delta_y
            if not (0 <= cell_x < width and 0 <= cell_y < height):
                patch[patch_row, patch_col, WALL_CHANNEL] = 1.0
                continue
            if (cell_x, cell_y) in other_player_positions:
                patch[patch_row, patch_col, WALL_CHANNEL] = 1.0
                continue
            tile_char = terrain[cell_y][cell_x]
            if tile_char in TILE_CHANNELS:
                patch[patch_row, patch_col, TILE_CHANNELS[tile_char]] = 1.0
    return patch


class EgocentricEncoder:
    """Convierte un estado de Overcooked en un vector plano egocéntrico híbrido de tamaño fijo."""

    observation_type = "semantic"

    def __init__(self, patch_radius: int = 1):
        """Inicializa el codificador.

        `patch_radius` es el radio del parche local; su lado es `2 * patch_radius + 1`
        (por defecto 1 -> parche 3x3, que ve solo los vecinos inmediatos). Se puede subir si un
        nivel necesita más contexto local, ya que la orientación global la dan los vectores.
        """
        self.patch_radius = int(patch_radius)
        self.patch_side = 2 * self.patch_radius + 1

    @property
    def observation_dim(self) -> int:
        """Devuelve la longitud fija del vector de observación."""
        patch_size = self.patch_side * self.patch_side * NUM_MAP_CHANNELS
        relative_size = len(OBJECT_TYPES) * VALUES_PER_OBJECT
        return patch_size + relative_size + NUM_ORIENTATIONS + len(HELD_OBJECT_CATEGORIES)

    def encode(self, state, mdp, agent_index: int = 0) -> np.ndarray:
        """Codifica el estado en el vector de observación del jugador `agent_index`."""
        player = state.players[agent_index]
        player_position = tuple(player.position)

        patch = self._encode_local_patch(state, mdp, agent_index, player_position)
        relative = self._encode_relative_object_vectors(state, mdp, agent_index, player_position)
        orientation = self._encode_orientation(player.orientation)
        held = self._encode_held_object(player.held_object)

        return np.concatenate([patch.reshape(-1), relative, orientation, held]).astype(np.float32)

    def _encode_local_patch(self, state, mdp, agent_index: int, player_position: tuple[int, int]) -> np.ndarray:
        """Construye el parche local one-hot centrado en el jugador (delega en `build_local_patch`)."""
        return build_local_patch(state, mdp, agent_index, player_position, self.patch_radius)

    def _encode_relative_object_vectors(self, state, mdp, agent_index: int, player_position: tuple[int, int]) -> np.ndarray:
        """Calcula (dx, dy, presencia) hacia el objeto más cercano de cada tipo.

        Para los objetos con los que se interactúa (cebolla, horno, plato, salida) el vector es la
        dirección del SIGUIENTE PASO del camino más corto real, calculado con BFS rodeando paredes (y
        al otro jugador, tratado como celda bloqueada). Así el agente sabe hacia dónde ir aunque haya
        muros de por medio. Para `other_player` se usa la dirección en línea recta (es coordinación,
        no navegación para interactuar). `presencia` es 1 si el objeto existe en el nivel.
        """
        terrain = mdp.terrain_mtx
        height = len(terrain)
        width = len(terrain[0]) if height > 0 else 0
        player_x, player_y = player_position

        other_player_positions = {
            tuple(other.position) for index, other in enumerate(state.players) if index != agent_index
        }
        # Un solo BFS desde el jugador; se reutiliza para todos los objetos con los que se interactúa.
        walkable = pathfinding.walkable_cells(mdp)
        distances, parents = pathfinding.bfs_from(player_position, walkable, set(other_player_positions))

        vectors = np.zeros(len(OBJECT_TYPES) * VALUES_PER_OBJECT, dtype=np.float32)
        for object_index, object_type in enumerate(OBJECT_TYPES):
            positions = self._object_positions(object_type, state, mdp, agent_index)
            base = object_index * VALUES_PER_OBJECT
            if not positions:
                continue  # deja (0, 0, 0): objeto ausente en este nivel

            if object_type == "other_player":
                # Dirección en línea recta al otro jugador (normalizada por el tamaño del mapa).
                nearest = self._nearest_position(player_position, positions)
                vectors[base + 0] = (nearest[0] - player_x) / max(width, 1)
                vectors[base + 1] = (nearest[1] - player_y) / max(height, 1)
                vectors[base + 2] = 1.0
                continue

            # Objetos interactuables: dirección del siguiente paso del camino más corto (BFS).
            vectors[base + 2] = 1.0  # el objeto existe en el nivel
            direction = pathfinding.next_step_direction(player_position, positions, distances, parents)
            if direction is not None:
                vectors[base + 0] = float(direction[0])
                vectors[base + 1] = float(direction[1])
        return vectors

    def _object_positions(self, object_type: str, state, mdp, agent_index: int) -> list[tuple[int, int]]:
        """Devuelve las posiciones de todos los objetos de un tipo dado en el nivel."""
        if object_type == "onion":
            return list(mdp.get_onion_dispenser_locations())
        if object_type == "pot":
            return list(mdp.get_pot_locations())
        if object_type == "dish":
            return list(mdp.get_dish_dispenser_locations())
        if object_type == "serve":
            return list(mdp.get_serving_locations())
        if object_type == "other_player":
            return [tuple(other.position) for index, other in enumerate(state.players) if index != agent_index]
        raise ValueError(f"Tipo de objeto desconocido: {object_type}")

    @staticmethod
    def _nearest_position(origin: tuple[int, int], positions: list[tuple[int, int]]) -> tuple[int, int] | None:
        """Devuelve la posición más cercana al origen por distancia Manhattan, o None si no hay."""
        if not positions:
            return None
        return min(positions, key=lambda position: abs(position[0] - origin[0]) + abs(position[1] - origin[1]))

    def _encode_orientation(self, orientation) -> np.ndarray:
        """Devuelve el one-hot de la orientación del jugador (NORTH/SOUTH/EAST/WEST)."""
        vector = np.zeros(NUM_ORIENTATIONS, dtype=np.float32)
        vector[Direction.DIRECTION_TO_INDEX[tuple(orientation)]] = 1.0
        return vector

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
