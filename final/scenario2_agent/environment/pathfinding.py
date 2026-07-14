"""Navegación consciente de paredes por BFS (búsqueda en anchura).

Da la dirección del SIGUIENTE PASO del camino más corto real hacia un objeto, rodeando paredes (y,
en la fase cooperativa, al otro jugador tratándolo como celda bloqueada). Se recalcula en cada
step, así que se adapta a obstáculos que se mueven. Sobre grids pequeños es muy barato.

Convención: las posiciones son (x, y). La caminabilidad se toma del terreno del MDP: solo se puede
pisar las celdas vacías (' ').
"""

from __future__ import annotations

from collections import deque

# Pasos de 4-conectividad en (dx, dy): norte, sur, este, oeste.
_STEPS = [(0, -1), (0, 1), (1, 0), (-1, 0)]


def walkable_cells(mdp) -> set[tuple[int, int]]:
    """Devuelve el conjunto de celdas caminables (tiles vacíos ' ') del terreno del MDP."""
    cells: set[tuple[int, int]] = set()
    for y, row in enumerate(mdp.terrain_mtx):
        for x, tile in enumerate(row):
            if tile == " ":
                cells.add((x, y))
    return cells


def bfs_from(start: tuple[int, int], walkable: set, blocked: set) -> tuple[dict, dict]:
    """BFS desde `start` sobre celdas caminables (menos las bloqueadas). Devuelve (distancias, padres).

    `distancias[cell]` es la distancia en pasos desde `start`; `padres[cell]` es la celda previa en
    el camino más corto (para reconstruir la ruta). Solo aparecen las celdas alcanzables.
    """
    distances = {start: 0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        current_x, current_y = current
        for delta_x, delta_y in _STEPS:
            neighbor = (current_x + delta_x, current_y + delta_y)
            if neighbor in walkable and neighbor not in blocked and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                parents[neighbor] = current
                queue.append(neighbor)
    return distances, parents


def first_step_towards(parents: dict, start: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int]:
    """Reconstruye el primer paso (dx, dy) del camino `start`->`goal` usando el mapa de padres."""
    node = goal
    previous = parents.get(node)
    while previous is not None and previous != start:
        node = previous
        previous = parents.get(node)
    return (node[0] - start[0], node[1] - start[1])


def next_step_direction(
    start: tuple[int, int],
    target_cells: list[tuple[int, int]],
    distances: dict,
    parents: dict,
) -> tuple[int, int] | None:
    """Dirección (dx, dy) del siguiente paso hacia el objeto alcanzable más cercano por camino real.

    `target_cells` son las celdas de los objetos (no caminables: dispensadores, hornos, salida). Como
    se interactúa desde una celda adyacente, el objetivo del BFS son las celdas caminables contiguas
    a un objeto. Si el jugador ya está adyacente a un objeto, devuelve la dirección para MIRARLO (y
    poder interactuar). Devuelve None si no hay objetos o ninguno es alcanzable.
    """
    if not target_cells:
        return None

    # Si ya está adyacente a un objeto, la mejor "siguiente acción" es girarse a mirarlo.
    adjacent_targets = [cell for cell in target_cells if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) == 1]
    if adjacent_targets:
        target = min(adjacent_targets)
        return (target[0] - start[0], target[1] - start[1])

    # Si no, buscar la celda caminable adyacente a un objeto que esté más cerca por BFS.
    best_goal = None
    best_distance = None
    for target in target_cells:
        for delta_x, delta_y in _STEPS:
            neighbor = (target[0] + delta_x, target[1] + delta_y)
            if neighbor in distances and (best_distance is None or distances[neighbor] < best_distance):
                best_distance = distances[neighbor]
                best_goal = neighbor
    if best_goal is None:
        return None
    return first_step_towards(parents, start, best_goal)
