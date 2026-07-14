"""Override de acción scripted para los interacts CATASTRÓFICOS/IRREVERSIBLES (no aprendidos).

Filosofía del híbrido: el override existe SOLO para evitar errores que arruinan el episodio y que no
tiene sentido dejar que la política reaprenda a cada rato. Todo lo demás (navegar, cuándo recoger,
etc.) lo aprende la política. Aquí se scriptean tres cosas:

1. ACTIVACIÓN DEL HORNO (irreversible): activar la olla a medio llenar (interact en vacío con <3
   cebollas) enciende una sopa incompleta y ya no admite más cebollas. Por eso:
   - manos vacías de cara a una olla LLENA sin cocinar -> se fuerza INTERACT (activar).
   - manos vacías de cara a una olla a MEDIAS -> se BLOQUEA el INTERACT (evita cocción prematura).

2. GUARD DE COORDINACIÓN (evita soltar/atascarse). Con un compañero en la cocina el mundo cambia
   entre que decides a dónde ir y que llegas: el otro llena la olla o saca la sopa antes que tú. Si al
   llegar interactúas igual, o SUELTAS el objeto en una mesa (se pierde) o te quedas atascado con un
   interact que ya no aplica. Regla: si llevas algo y quieres INTERACT pero la casilla que encaras NO
   admite ese objeto ahora mismo, en vez de interactuar te HACES A UN LADO (mueves a una casilla libre)
   y reintentas al tick siguiente; así el compañero termina lo suyo y no os estorbáis.

   Qué interact es "productivo" según lo que llevas:
   - cebolla -> solo sobre una olla ADDABLE (con hueco y sin cocinar).
   - plato   -> sobre una olla LISTA (recoger) o COCINANDO/llena (esperar a que esté lista).
   - sopa    -> solo sobre una casilla de SALIDA (entregar).

3. SOLTAR TRAS ATASCO (re-planear). Si el objeto en mano lleva `drop_after_stuck_ticks` ticks SIN
   ningún destino útil (p. ej. cebolla y todas las ollas ocupadas, o plato y ninguna olla lista ni
   cocinando porque el compañero ya sacó la sopa), el agente deja de esperar: va a una MESA LIBRE y
   suelta ahí el objeto para quedar con las manos vacías y volver a planear (ir por un plato / una
   cebolla, lo que toque). Soltar en una mesa vacía es recuperable (se puede recoger luego y los
   dispensadores son infinitos), a diferencia de soltarlo en cualquier sitio por un mal interact.

Se aplica igual en entrenamiento (como parte del entorno) y en inferencia (watch / juego en pareja).
"""

from __future__ import annotations

from scenario2_agent.environment import pathfinding
from scenario2_agent.environment.actions import INTERACT_ACTION_INDEX

# Dirección (dx, dy) -> índice de acción de movimiento (N, S, E, W). Debe coincidir con AGENT_ACTIONS.
_DIRECTION_TO_ACTION = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}


def make_pot_activation_override(target_provider, onions_target: int = 3, drop_after_stuck_ticks: int = 5):
    """Devuelve el override scripted: horno seguro + guard de coordinación + soltar tras atasco.

    `drop_after_stuck_ticks` es cuántos ticks debe llevar el objeto en mano SIN destino útil antes de
    ir a soltarlo en una mesa libre y re-planear. El contador de atasco es por jugador (`agent_index`),
    de modo que el mismo override sirve para los dos agentes de una partida en pareja.
    """
    stuck_ticks_by_agent: dict[int, int] = {}

    def override(action_index: int, state, mdp, agent_index: int = 0) -> int:
        player = state.players[agent_index]
        faced_cell = _faced_cell(player)
        pot_locations = set(mdp.get_pot_locations())
        held_object = player.held_object

        # --- Con objeto en mano: guard de coordinación + soltar tras atasco ---
        if held_object is not None:
            # Contador de "sin destino útil": se reinicia en cuanto vuelve a haber a dónde ir.
            if _held_has_destination(held_object, state, mdp, pot_locations):
                stuck_ticks_by_agent[agent_index] = 0
            else:
                stuck_ticks_by_agent[agent_index] = stuck_ticks_by_agent.get(agent_index, 0) + 1

            # Atascado demasiado tiempo: ir a soltar el objeto en una mesa libre y re-planear.
            if stuck_ticks_by_agent.get(agent_index, 0) >= drop_after_stuck_ticks:
                return _go_drop_object(state, mdp, agent_index, faced_cell)

            # Guard normal: si quiere INTERACT pero la casilla encarada no admite el objeto, apartarse.
            wants_interact = int(action_index) == INTERACT_ACTION_INDEX
            if wants_interact and not _interact_is_productive(held_object, faced_cell, state, mdp, pot_locations):
                return _step_aside(state, mdp, agent_index, faced_cell)
            return action_index

        # --- Manos vacías: activación segura del horno ---
        stuck_ticks_by_agent[agent_index] = 0
        if faced_cell not in pot_locations:
            return action_index
        soup = state.get_object(faced_cell) if state.has_object(faced_cell) else None
        if soup is not None and (soup.is_cooking or soup.is_ready):
            return action_index  # ya cocina/está lista: nada que forzar
        onions = len(soup.ingredients) if soup is not None else 0
        if onions >= onions_target:
            return INTERACT_ACTION_INDEX  # olla llena y de cara: activar el horno
        if onions > 0 and int(action_index) == INTERACT_ACTION_INDEX:
            # Olla a medias: bloquear el interact (evita cocción prematura irreversible).
            return _safe_move_towards_target(state, mdp, agent_index, target_provider)
        return action_index

    return override


def _faced_cell(player) -> tuple[int, int]:
    """Devuelve la celda que el jugador tiene enfrente (posición + orientación)."""
    orientation_x, orientation_y = player.orientation
    return (player.position[0] + orientation_x, player.position[1] + orientation_y)


def _held_has_destination(held_object, state, mdp, pot_locations) -> bool:
    """Indica si el objeto en mano tiene AHORA algún sitio útil a donde ir (destino existe en el mapa).

    Es una comprobación global (no de la casilla encarada): sirve para decidir si el agente está
    realmente atascado con el objeto o solo le falta llegar/girar. La sopa siempre tiene destino
    (la salida), así que nunca se considera atascada (nunca se suelta una sopa terminada).
    """
    name = held_object.name
    if name == "onion":
        return any(_pot_is_addable(state, position) for position in pot_locations)
    if name == "dish":
        return any(_pot_is_soup_coming(state, position) for position in pot_locations)
    if name == "soup":
        return len(mdp.get_serving_locations()) > 0
    return True  # objeto desconocido: no forzar nada


def _pot_is_addable(state, pot_position) -> bool:
    """La olla admite otra cebolla: vacía, o con hueco y aún sin llenar/cocinar/lista."""
    if not state.has_object(pot_position):
        return True
    soup = state.get_object(pot_position)
    return not (soup.is_cooking or soup.is_ready or soup.is_full)


def _pot_is_soup_coming(state, pot_position) -> bool:
    """La olla dará sopa pronto o ya la tiene: lista, cocinando o llena (para justificar llevar plato)."""
    if not state.has_object(pot_position):
        return False
    soup = state.get_object(pot_position)
    return soup.is_ready or soup.is_cooking or soup.is_full


def _interact_is_productive(held_object, faced_cell, state, mdp, pot_locations) -> bool:
    """Indica si interactuar AHORA con la casilla encarada sirve para el objeto que se lleva.

    'Productivo' incluye el caso de ESPERAR: llevar un plato de cara a una olla que aún cocina se
    considera productivo (interact es un no-op inofensivo y conviene quedarse hasta que esté lista).
    Devuelve False cuando el interact solo soltaría el objeto o no aplica (hay que hacerse a un lado).
    """
    name = held_object.name
    if name == "onion":
        return faced_cell in pot_locations and _pot_is_addable(state, faced_cell)
    if name == "dish":
        return faced_cell in pot_locations and _pot_is_soup_coming(state, faced_cell)
    if name == "soup":
        return faced_cell in set(mdp.get_serving_locations())
    return True  # objeto desconocido: no intervenir


def _go_drop_object(state, mdp, agent_index: int, faced_cell) -> int:
    """Devuelve la acción para ir a soltar el objeto en la MESA LIBRE alcanzable más cercana.

    Si ya se encara una mesa libre -> INTERACT (soltar ahí). Si no, se navega hacia ella (BFS). Si no
    hay ninguna mesa libre, se hace a un lado (esperar) en vez de soltarlo en un sitio inválido.
    """
    empty_counters = list(mdp.get_empty_counter_locations(state))
    if not empty_counters:
        return _step_aside(state, mdp, agent_index, faced_cell)
    if faced_cell in set(empty_counters):
        return INTERACT_ACTION_INDEX  # de cara a una mesa libre: soltar el objeto

    player = state.players[agent_index]
    player_position = tuple(player.position)
    walkable = pathfinding.walkable_cells(mdp)
    other_players = {tuple(other.position) for index, other in enumerate(state.players) if index != agent_index}
    distances, parents = pathfinding.bfs_from(player_position, walkable, other_players)
    direction = pathfinding.next_step_direction(player_position, empty_counters, distances, parents)
    if direction in _DIRECTION_TO_ACTION:
        return _DIRECTION_TO_ACTION[direction]
    return _step_aside(state, mdp, agent_index, faced_cell)


def _step_aside(state, mdp, agent_index: int, faced_cell) -> int:
    """Devuelve un MOVIMIENTO hacia una casilla adyacente libre para apartarse (nunca INTERACT).

    Evita la celda encarada (el objetivo que ya no aplica) y la casilla del otro jugador, para no
    estorbarse. Si no hay ninguna libre, gira hacia otro lado (moverse aunque choque cambia la
    orientación y, sobre todo, NO suelta el objeto).
    """
    player = state.players[agent_index]
    player_x, player_y = player.position
    walkable = pathfinding.walkable_cells(mdp)
    other_players = {tuple(other.position) for index, other in enumerate(state.players) if index != agent_index}

    for (delta_x, delta_y), action in _DIRECTION_TO_ACTION.items():
        neighbor = (player_x + delta_x, player_y + delta_y)
        if neighbor == faced_cell or neighbor in other_players:
            continue
        if neighbor in walkable:
            return action
    # Sin vecino libre: cualquier movimiento que no sea hacia la celda encarada (no suelta el objeto).
    for (delta_x, delta_y), action in _DIRECTION_TO_ACTION.items():
        if (player_x + delta_x, player_y + delta_y) != faced_cell:
            return action
    return 0


def _safe_move_towards_target(state, mdp, agent_index: int, target_provider) -> int:
    """Devuelve una acción de MOVIMIENTO hacia el objetivo del controlador (nunca INTERACT).

    Sustituye a un INTERACT que cocinaría prematuramente: aleja al jugador de la olla hacia donde toca.
    Si no hay dirección clara, devuelve un movimiento cualquiera (en el peor caso choca, no cocina).
    """
    player = state.players[agent_index]
    player_position = tuple(player.position)
    targets = list(target_provider(state, mdp, agent_index))
    if targets:
        walkable = pathfinding.walkable_cells(mdp)
        other_players = {tuple(other.position) for index, other in enumerate(state.players) if index != agent_index}
        distances, parents = pathfinding.bfs_from(player_position, walkable, other_players)
        direction = pathfinding.next_step_direction(player_position, targets, distances, parents)
        if direction in _DIRECTION_TO_ACTION:
            return _DIRECTION_TO_ACTION[direction]
    return 0  # sin dirección: moverse (N); nunca interactuar
