"""Controladores de objetivo (la "receta" de alto nivel), por stage.

Un `target_provider` es una función `f(state, mdp, agent_index) -> [celdas]` que decide A QUÉ SITIO
debe dirigirse el agente AHORA mismo, según el estado. Es el nivel alto (guionado y ajustable) de la
arquitectura jerárquica: el `GuidedEncoder` le pregunta el objetivo, calcula por BFS la dirección y la
distancia hacia él, y el MLP solo aprende a seguir esa recomendación e interactuar al llegar.

Aquí vive la lógica semántica que antes queríamos que el agente descubriera solo (qué objeto importa
en cada momento). Al ser código, se puede depurar y afinar; y en la fase cooperativa (Parte 2) es
donde meteremos la conciencia del compañero.
"""

from __future__ import annotations


def onion_target(state, mdp, agent_index: int = 0) -> list[tuple[int, int]]:
    """P1: el objetivo es siempre un dispensador de cebolla."""
    return list(mdp.get_onion_dispenser_locations())


def onion_then_pot_target(state, mdp, agent_index: int = 0) -> list[tuple[int, int]]:
    """P2: si el jugador NO lleva cebolla, el objetivo es la cebolla; si la lleva, el horno.

    Así el mismo controlador guía las dos mitades de la tarea (recoger y colocar). Para el objetivo de
    llenar la olla con varias cebollas, el ciclo se repite solo: tras colocar una, el jugador queda con
    las manos vacías y el objetivo vuelve a ser la cebolla.
    """
    held = state.players[agent_index].held_object
    if held is not None and held.name == "onion":
        return list(mdp.get_pot_locations())
    return list(mdp.get_onion_dispenser_locations())


def _classify_pots(state, mdp):
    """Clasifica las ollas del nivel según lo que toca hacer con ellas.

    Devuelve tres listas de posiciones:
    - `addable`: se les puede echar una cebolla (vacías o con hueco y aún sin cocinar).
    - `full_idle`: llenas pero SIN cocinar todavía (hay que activarlas con un interact en vacío).
    - `busy`: ya cocinando o listas (no hay que tocarlas para llenar/activar).
    """
    addable, full_idle, busy = [], [], []
    for pot_position in mdp.get_pot_locations():
        if not state.has_object(pot_position):
            addable.append(pot_position)
            continue
        soup = state.get_object(pot_position)
        if soup.is_cooking or soup.is_ready:
            busy.append(pot_position)
        elif soup.is_full:
            full_idle.append(pot_position)
        else:
            addable.append(pot_position)
    return addable, full_idle, busy


def fill_and_cook_target(state, mdp, agent_index: int = 0) -> list[tuple[int, int]]:
    """P2 con cocción: llenar la olla con cebollas y luego activarla (empezar a cocinar).

    Lógica del controlador:
    - Si lleva una cebolla -> objetivo = una olla con hueco (para echarla).
    - Manos vacías y hay una olla LLENA sin cocinar -> objetivo = esa olla (para activarla con interact).
    - Manos vacías y ninguna olla llena -> objetivo = la cebolla (ir a por la siguiente).
    """
    addable, full_idle, _ = _classify_pots(state, mdp)
    held = state.players[agent_index].held_object
    if held is not None and held.name == "onion":
        return addable or list(mdp.get_pot_locations())
    if full_idle:
        return full_idle
    return list(mdp.get_onion_dispenser_locations())


def full_soup_target(state, mdp, agent_index: int = 0) -> list[tuple[int, int]]:
    """Receta completa: llenar la olla -> cocinar -> coger plato -> sacar la sopa -> entregarla.

    El controlador decide el objetivo según lo que lleva y el estado de las ollas:
    - lleva SOPA -> estación de entrega (serving).
    - lleva PLATO -> la olla (a sacar la sopa; espera ahí si aún no está lista).
    - lleva CEBOLLA -> una olla con hueco.
    - manos vacías:
        * hay una olla cocinando o lista -> dispensador de PLATOS (prepararse para sacarla).
        * hay una olla llena sin cocinar -> esa olla (activar; lo remata el override del horno).
        * si no -> dispensador de CEBOLLAS (seguir llenando).
    """
    player = state.players[agent_index]
    held = player.held_object
    if held is not None and held.name == "soup":
        return list(mdp.get_serving_locations())
    if held is not None and held.name == "dish":
        return list(mdp.get_pot_locations())
    if held is not None and held.name == "onion":
        addable, _full_idle, _busy = _classify_pots(state, mdp)
        return addable or list(mdp.get_pot_locations())
    addable, full_idle, busy = _classify_pots(state, mdp)
    if busy:
        return list(mdp.get_dish_dispenser_locations())
    if full_idle:
        return full_idle
    return list(mdp.get_onion_dispenser_locations())


# Registro por stage: el nombre del stage -> su controlador de objetivo.
TARGET_PROVIDERS = {
    "p1": onion_target,
    "p2": onion_then_pot_target,
}


def select_target_provider(stage: str, cook: bool = False, serve: bool = False):
    """Elige el controlador de objetivo según el stage y la tarea (cocinar / receta completa)."""
    if stage == "p1":
        return onion_target
    if serve:
        return full_soup_target
    if cook:
        return fill_and_cook_target
    return onion_then_pot_target
