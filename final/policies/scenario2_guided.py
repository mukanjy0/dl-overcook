"""Política del estudiante: adapta nuestro agente PPO (observación guiada) al contrato del profesor.

El evaluador espera una clase `StudentAgent` con:

    __init__(self, config: dict)
    reset(self)
    act(self, obs) -> int

Convención de acciones del profesor (6): 0 norte, 1 sur, 2 este, 3 oeste, 4 quedarse, 5 interactuar.
Nuestro modelo usa 5 acciones (0-3 movimiento, 4 interactuar; SIN "quedarse"); se mapea nuestro 4 -> 5.

La firma `act(obs)` es fija, pero lo que hacemos dentro es nuestro: el modelo NO consume el vector
featurizado del profesor, sino que reconstruye su observación GUIADA a partir del ESTADO crudo + el
MDP. Por eso el run debe usar `observation.type: state` (que hace `obs = {"state","mdp","agent_index"}`).

Autocontenido: este archivo bundlea el paquete `scenario2_agent` (carpeta hermana) y el checkpoint
`scenario2_guided_model.pt` (junto a este archivo). Ambos se localizan de forma relativa a este archivo, así que la
submission funciona sin importar desde qué carpeta se ejecute el evaluador.
"""

from __future__ import annotations

import pathlib
import sys

# --- Localizar dependencias bundle de forma relativa a este archivo ---
_HERE = pathlib.Path(__file__).resolve().parent            # .../final/policies
_BUNDLE_ROOT = _HERE.parent                                # .../final  (contiene scenario2_agent/)
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

import scenario2_agent  # noqa: F401  (aplica el shim de compatibilidad de NumPy al importarse)
from scenario2_agent.control.pot_activation import make_pot_activation_override
from scenario2_agent.observations.guided_encoder import GuidedEncoder
from scenario2_agent.observations.target_providers import select_target_provider
from scenario2_agent.storage.checkpoints import load_checkpoint

# Checkpoint por defecto: bundled junto a este archivo.
_DEFAULT_MODEL = _HERE / "scenario2_guided_model.pt"

# Nuestra acción -> acción del profesor. 0-3 (movimiento) coinciden; nuestro 4 (interactuar) -> 5.
_OUR_TO_PROFESSOR_ACTION = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}

# Caché de modelos por ruta: el evaluador reconstruye la política por intento y no queremos releer el .pt.
_MODEL_CACHE: dict = {}


def _resolve_checkpoint(config: dict) -> str:
    """Determina la ruta del checkpoint: config['model_path'] o config['checkpoint'], si no el bundled."""
    candidate = config.get("model_path") or config.get("checkpoint")
    if not candidate:
        return str(_DEFAULT_MODEL)
    path = pathlib.Path(candidate)
    if not path.is_absolute():
        # Relativo: probar respecto al cwd y, si no existe, respecto al bundle.
        if not path.exists() and (_BUNDLE_ROOT / path).exists():
            path = _BUNDLE_ROOT / path
    return str(path)


def _load_model_cached(checkpoint_path: str):
    """Carga (y cachea) el modelo y su payload desde un checkpoint, por ruta resuelta."""
    resolved = str(pathlib.Path(checkpoint_path).resolve())
    if resolved not in _MODEL_CACHE:
        _MODEL_CACHE[resolved] = load_checkpoint(resolved)
    return _MODEL_CACHE[resolved]


class StudentAgent:
    """Envuelve nuestro modelo PPO (observación guiada + override) tras el contrato del profesor."""

    def __init__(self, config=None):
        """Carga el checkpoint y arma el encoder guiado y el override una sola vez (no en cada paso).

        Config admitida (inline o vía model_path/config_path del evaluador):
          - model_path / checkpoint: ruta al .pt (default: bundled Scenario 2 model).
          - stochastic: muestrear la acción (True, recomendado) o argmax (False).
          - stage: 'p3' (receta completa, default) o 'p2'.
          - drop_after: ticks con un objeto sin destino útil antes de ir a soltarlo (default 5).
        """
        self.config = config or {}
        self.stochastic = bool(self.config.get("stochastic", True))
        self.stage = str(self.config.get("stage", "p3"))
        self.drop_after = int(self.config.get("drop_after", 5))

        import torch  # import diferido: solo se necesita para inferencia

        self._torch = torch
        seed = self.config.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
        self.model, payload = _load_model_cached(_resolve_checkpoint(self.config))

        serving = self.stage == "p3"
        self.onions_target = 3 if serving else int(self.config.get("onions_to_pot", 3))
        self.target_provider = select_target_provider(self.stage, cook=True, serve=serving)

        encoder_meta = payload["meta"]["encoder"]
        patch_radius = int(encoder_meta["patch_radius"])
        if encoder_meta.get("type", "semantic") != "guided":
            raise ValueError("Este StudentAgent espera un checkpoint entrenado con observación 'guided'.")
        self.encoder = GuidedEncoder(self.target_provider, patch_radius=patch_radius)
        expected_dim = int(payload["meta"]["model"]["input_dim"])
        if self.encoder.observation_dim != expected_dim:
            raise ValueError(
                f"La observación guiada ({self.encoder.observation_dim}) no coincide con el checkpoint ({expected_dim})."
            )

        self.reset()

    def reset(self):
        """Reinicia el estado por episodio: recrea el override para limpiar el contador de atasco."""
        self.action_override = make_pot_activation_override(
            self.target_provider, self.onions_target, drop_after_stuck_ticks=self.drop_after
        )

    def act(self, obs) -> int:
        """Devuelve una acción del profesor (int en 0..5) a partir del estado crudo + mdp del `obs`.

        `obs` debe ser el dict de `observation.type: state`: {"state", "mdp", "agent_index"}.
        """
        state, mdp, agent_index = _unpack_state_obs(obs)

        observation = self.encoder.encode(state, mdp, agent_index=agent_index)
        observation_tensor = self._torch.as_tensor(observation, dtype=self._torch.float32).unsqueeze(0)
        action, _log_probability, _value = self.model.act(observation_tensor, deterministic=not self.stochastic)
        our_action_index = int(action.item())

        # Override scripted (horno seguro + guard de coordinación con el compañero).
        our_action_index = self.action_override(our_action_index, state, mdp, agent_index)
        return _OUR_TO_PROFESSOR_ACTION[our_action_index]


def _unpack_state_obs(obs) -> tuple:
    """Extrae (state, mdp, agent_index) del `obs`; exige `observation.type: state`.

    Da un error claro si llega otro formato (p. ej. el vector featurizado), porque nuestro modelo
    necesita el estado y el mdp para reconstruir su observación guiada.
    """
    if not isinstance(obs, dict) or "state" not in obs or "mdp" not in obs:
        raise ValueError(
            "StudentAgent necesita observation.type: 'state' (obs con 'state' y 'mdp'). "
            "Configura el competition.yaml con observation: {type: state}."
        )
    agent_index = int(obs.get("agent_index", 0))
    return obs["state"], obs["mdp"], agent_index
