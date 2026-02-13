"""
Paquete de utilidades para análisis de clustering y regresión de videojuegos.

Módulos:
    - data_preprocessing: Funciones para carga y limpieza de datos
    - clustering: Implementación de K-Means y PCA
    - regression: Modelos de regresión por cluster
    - visualization: Funciones de visualización interactivas
"""

__version__ = '1.0.0'
__author__ = 'nicolarus05'

from . import data_preprocessing
from . import clustering
from . import regression
from . import visualization

__all__ = [
    'data_preprocessing',
    'clustering',
    'regression',
    'visualization'
]
