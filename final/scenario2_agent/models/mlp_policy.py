"""Red MLP política + valor, escrita desde cero para PPO.

Son DOS perceptrones multicapa INDEPENDIENTES (no comparten cuerpo):

- Red POLÍTICA: produce `num_actions` logits. Al aplicar softmax se obtiene la distribución sobre
  acciones de la que el agente muestrea (o toma el argmax en evaluación). En este proyecto hay 5
  acciones: arriba, abajo, derecha, izquierda e interactuar (sin "quedarse quieto", para forzar que
  el agente no se quede parado).
- Red de VALOR (critic): estima el valor del estado, que PPO necesita para calcular las ventajas.
  No se usa al jugar, solo durante el entrenamiento.

Por qué SEPARADAS y no un cuerpo compartido: si comparten cuerpo, un error grande del critic (p. ej.
al reusar el modelo en una tarea con recompensas de otra escala) mete gradientes enormes que
destrozan las features de la política y colapsan el entrenamiento. Con redes independientes, el
critic puede recalibrarse sin dañar la política. Cuesta un poco más de cómputo (dos MLPs pequeños),
despreciable aquí.

Sobre el dropout: se deja configurable pero por defecto en 0 para el track PPO. El dropout añade
ruido aleatorio distinto en cada forward, lo que en PPO hace inconsistentes las probabilidades
entre la recolección (rollout) y la actualización, y puede desestabilizar el clip. Es más natural
en el track genético (para diversidad de neuronas). Si se activa aquí, conviene mantenerlo bajo.
"""

from __future__ import annotations

import torch
from torch import nn

# 5 acciones: 0 arriba (N), 1 abajo (S), 2 derecha (E), 3 izquierda (W), 4 interactuar. Sin STAY.
NUM_ACTIONS = 5


def _build_mlp_body(input_dim: int, hidden_sizes: tuple[int, ...], activation: str, dropout: float) -> tuple[nn.Sequential, int]:
    """Construye un cuerpo MLP (capas ocultas) y devuelve (módulo, tamaño de salida)."""
    activation_layer = nn.Tanh if activation == "tanh" else nn.ReLU
    layers: list[nn.Module] = []
    previous_size = int(input_dim)
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(previous_size, int(hidden_size)))
        layers.append(activation_layer())
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous_size = int(hidden_size)
    return nn.Sequential(*layers), previous_size


class MLPPolicyValue(nn.Module):
    """Dos MLPs independientes: uno de política y uno de valor (critic)."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int = NUM_ACTIONS,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        """Construye las dos redes.

        `input_dim` es el tamaño del vector de observación. `hidden_sizes` define el cuerpo de CADA
        red, `activation` la no linealidad ('tanh' o 'relu'), y `dropout` la probabilidad de apagado
        de neuronas tras cada capa oculta (0 = desactivado).
        """
        super().__init__()
        self.policy_body, policy_output_size = _build_mlp_body(input_dim, hidden_sizes, activation, dropout)
        self.value_body, value_output_size = _build_mlp_body(input_dim, hidden_sizes, activation, dropout)
        self.policy_head = nn.Linear(policy_output_size, int(num_actions))
        self.value_head = nn.Linear(value_output_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (logits de política, valor), cada uno de su propia red."""
        observations = observations.float()
        logits = self.policy_head(self.policy_body(observations))
        value = self.value_head(self.value_body(observations)).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Elige acciones para actuar en el entorno.

        Devuelve (acción, log-probabilidad, valor). Si `deterministic` es True toma el argmax
        (para evaluar/ver jugar); si no, muestrea de la distribución (para explorar en el rollout).
        """
        logits, value = self.forward(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action)
        return action, log_probability, value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reevalúa acciones ya tomadas para la actualización de PPO.

        Devuelve (log-probabilidad de esas acciones bajo la política actual, entropía de la
        distribución, valor estimado). La entropía sirve como bonus de exploración en PPO.
        """
        logits, value = self.forward(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probabilities, entropy, value
