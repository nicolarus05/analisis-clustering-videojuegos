# Análisis de Clustering y Regresión - Video Game Sales

## 📋 Descripción del Proyecto

Proyecto de análisis de datos que implementa técnicas de **clustering** (K-Means) y **regresión** para segmentar videojuegos según sus patrones de ventas y predecir ventas globales por cluster.

El proyecto utiliza **Análisis de Componentes Principales (PCA)** para reducción dimensional y visualización, seguido de modelos de regresión específicos para cada segmento identificado.

## 🎮 Dataset

- **Nombre**: Video Game Sales Dataset
- **Fuente**: [Kaggle - Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)
- **Registros**: ~16,500 videojuegos (1980-2016)
- **Variables principales**:
  - `Name`: Nombre del videojuego
  - `Platform`: Plataforma (PS4, Xbox, PC, etc.)
  - `Year`: Año de lanzamiento
  - `Genre`: Género del juego
  - `Publisher`: Empresa publicadora
  - `NA_Sales`: Ventas en Norteamérica (millones)
  - `EU_Sales`: Ventas en Europa (millones)
  - `JP_Sales`: Ventas en Japón (millones)
  - `Other_Sales`: Ventas en otras regiones (millones)
  - `Global_Sales`: Ventas globales totales (millones)

## 🎯 Objetivos

1. **Segmentación**: Identificar grupos de videojuegos con patrones de ventas similares
2. **Reducción dimensional**: Aplicar PCA para visualización en 2D
3. **Predicción**: Desarrollar modelos de regresión por cluster para predecir ventas globales
4. **Análisis**: Caracterizar cada segmento y entender sus patrones de mercado

## 🛠️ Metodología

### Fase 1: Análisis Exploratorio (EDA)
- Análisis de distribuciones de ventas
- Identificación de valores nulos y outliers
- Análisis temporal de tendencias

### Fase 2: Preprocesamiento
- Limpieza de datos (valores nulos, duplicados)
- Selección de features relevantes
- Normalización con StandardScaler

### Fase 3: Clustering
- Aplicación del método del codo (Elbow Method)
- K-Means clustering con k óptimo
- Análisis de centroides

### Fase 4: PCA
- Reducción a 2-3 componentes principales
- Visualización interactiva de clusters
- Análisis de varianza explicada

### Fase 5: Regresión por Cluster
- Regresión Lineal por segmento
- Random Forest Regressor por segmento
- Evaluación con métricas R² y RMSE

## � Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/nicolarus05/analisis-clustering-videojuegos.git
   cd analisis-clustering-videojuegos
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Uso

1. Descarga el dataset de Kaggle y colócalo en la carpeta `data/raw/`.

2. Ejecuta los notebooks en orden:
   - `01_exploratory_analysis.ipynb`: Análisis exploratorio
   - `02_preprocessing.ipynb`: Preprocesamiento de datos
   - `03_clustering_analysis.ipynb`: Análisis de clustering
   - `04_regression_models.ipynb`: Modelos de regresión

3. Revisa los resultados en la carpeta `results/`.

## 📁 Estructura del Proyecto

```
analisis-clustering-videojuegos/
│
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Archivos a ignorar en Git
│
├── data/                        # Datasets
│   ├── raw/                     # Datos originales
│   │   └── vgsales.csv
│   └── processed/               # Datos procesados
│       ├── vgsales_clean.csv
│       └── vgsales_clustered.csv
│
├── notebooks/                   # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_regression_models.ipynb
│
├── src/                         # Scripts Python
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── clustering.py
│   ├── regression.py
│   └── visualization.py
│
├── results/                     # Resultados
│   ├── figures/                 # Gráficos y visualizaciones
│   │   ├── elbow_method.png
│   │   ├── pca_clusters.html
│   │   ├── cluster_distribution.png
│   │   └── regression_performance.png
│   └── models/                  # Resultados de modelos
│       └── regression_results.csv
│
└── reports/                     # Informes finales
    └── analisis_final.pdf
```

## 📊 Resultados

- **Clusters identificados**: [Descripción breve de los clusters encontrados]
- **Métricas de regresión**: R² y RMSE por cluster
- **Visualizaciones**: Gráficos de PCA, distribuciones, etc.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

- **Nombre**: [Nicolás García Hernández]
- **GitHub**: [Tu GitHub](https://github.com/nicolarus05)

## 🚀 Instalación y Uso

### Prerequisitos
- Python 3.9 o superior
- pip instalado

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/nicolarus05/analisis-clustering-videojuegos.git
cd analisis-clustering-videojuegos

    Instalar dependencias:

bash
pip install -r requirements.txt

    Descargar el dataset:

        Descargar vgsales.csv desde Kaggle

        Colocar en la carpeta data/raw/

Ejecución

Ejecutar los notebooks en orden:

bash
jupyter notebook

    01_exploratory_analysis.ipynb - Análisis exploratorio inicial

    02_preprocessing.ipynb - Limpieza y preparación de datos

    03_clustering_analysis.ipynb - Clustering y PCA

    04_regression_models.ipynb - Modelos de regresión

📊 Resultados Esperados

    Número de clusters óptimo: 4-5 segmentos

    Varianza explicada (PCA): ~80-90% con 2 componentes

    Interpretación de clusters:

        Cluster 0: Blockbusters globales

        Cluster 1: Juegos populares en Occidente

        Cluster 2: Juegos de nicho japonés

        Cluster 3: Juegos de ventas moderadas

        Cluster 4: Juegos de bajo rendimiento

🔧 Tecnologías Utilizadas

    Python 3.9+

    Análisis de datos: pandas, numpy

    Machine Learning: scikit-learn

    Visualización: matplotlib, seaborn, plotly

    Notebooks: Jupyter

📈 Métricas de Evaluación

    Clustering: WCSS (Within-Cluster Sum of Squares), Silhouette Score

    Regresión: R² Score, RMSE (Root Mean Squared Error), MAE

👤 Autor

[Tu Nombre]

    GitHub: @nicolarus05

    Proyecto: Análisis de Clustering y Regresión

    Fecha: Febrero 2026

📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.
🙏 Agradecimientos

    Dataset proporcionado por Kaggle

    Inspirado en técnicas de segmentación de clientes aplicadas al mercado de videojuegos
