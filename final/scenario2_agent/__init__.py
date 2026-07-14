"""Paquete overcook_ppo: entrenamiento por stages (currículum) de un agente Overcooked con PPO desde cero.

Al importar este paquete se aplica un pequeño shim de compatibilidad de NumPy, porque
`overcooked_ai==1.1.0` usa alias que NumPy 2.0 eliminó (`np.Inf`, `np.int`). El shim se ejecuta
aquí, antes de que cualquier submódulo importe overcooked, para no tener que tocar el venv ni
cambiar la versión fijada de NumPy.
"""

from __future__ import annotations


def _apply_numpy_compatibility_shim() -> None:
    """Restaura los alias de NumPy eliminados en 2.0 que overcooked_ai 1.1.0 todavía usa.

    overcooked_ai 1.1.0 fue escrito para NumPy < 2.0 y referencia `np.Inf` y `np.int`, que se
    quitaron en NumPy 2.0. Recreamos esos alias apuntando a sus reemplazos oficiales para que la
    librería importe sin errores. Solo se añaden si faltan, de modo que en un NumPy < 2.0 esto no
    cambia nada.
    """
    import numpy

    if not hasattr(numpy, "Inf"):
        numpy.Inf = numpy.inf
    if not hasattr(numpy, "int"):
        numpy.int = int


_apply_numpy_compatibility_shim()
