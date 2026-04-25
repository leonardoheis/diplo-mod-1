# Trabajo Práctico Final — Machine Learning & Deep Learning

Trabajo práctico final del módulo de **Machine Learning & Deep Learning** de la diplomatura.

## Objetivo

Aplicar las técnicas vistas en la diplomatura a un problema real de datos, recorriendo el ciclo completo:

1. Exploración (EDA)
2. Preprocesamiento
3. Entrenamiento de modelos
4. Evaluación crítica de resultados

## Consigna

### Dataset

Utilizar un dataset propio del entorno laboral (recomendado) o uno de los datasets de ejemplo listados más abajo.

### Modelos a entrenar

Se entrenarán al menos **dos modelos** sobre el mismo dataset, aplicados al mismo problema:

- **Un modelo de ML clásico**: Regresión Lineal/Logística, Decision Tree, XGBoost, Random Forest, u otro justificado.
- **Una Red Neuronal**: con TensorFlow o PyTorch.

Esto permite comparar directamente un enfoque clásico contra uno de Deep Learning.

## Estructura del informe

1. **Introducción** — Descripción del dataset, variable objetivo y tipo de problema (clasificación o regresión).
2. **Análisis Exploratorio (EDA)** — Estadísticas descriptivas, visualizaciones, detección de desbalanceo, outliers y datos faltantes.
3. **Preprocesamiento** — Tratamiento de nulos y outliers, encoding de categóricas, normalización, split train/test, técnicas de balanceo.
4. **Entrenamiento**
   - *Modelo de ML*: justificación del algoritmo, hiperparámetros y tuning.
   - *Red Neuronal*: arquitectura (capas, activaciones), optimizador, epochs, curvas de entrenamiento, regularización (dropout, early stopping, etc.).
5. **Evaluación y comparación** — Sección clave del trabajo.
   - Clasificación: accuracy, precision, recall, F1-score, matriz de confusión (obligatoria para ambos modelos), ROC/AUC opcional.
   - Regresión: MAE, RMSE, R², gráfico de predicciones vs. valores reales.
   - Análisis crítico: ¿qué modelo performó mejor y por qué?, ¿la red neuronal superó al ML clásico?, ¿hay overfitting?, ¿qué features fueron más relevantes?, ¿qué se mejoraría con más datos/tiempo?
6. **Conclusiones** — Síntesis de resultados, lecciones aprendidas y trabajo futuro.

## Datasets de ejemplo

| Dataset | Tipo | Registros | Fuente |
| --- | --- | --- | --- |
| [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Clasificación | 7.043 | IBM / Kaggle |
| [Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45) | Clasificación | 303 | UCI ML Repository |
| California Housing (`sklearn.datasets.fetch_california_housing`) | Regresión | 20.640 | StatLib / scikit-learn |

## Estructura del repositorio

```
.
├── data/
│   ├── raw/             # Original, immutable data
│   ├── interim/         # Intermediate transformations
│   └── processed/       # Final, model-ready data
├── notebooks/           # Numbered notebooks (one per report section)
├── src/                 # Reusable source code (diplo_mod_1 package)
├── models/              # Trained models / checkpoints (not versioned)
├── reports/             # Final report and figures
├── pyproject.toml       # Dependencies and project metadata (uv)
├── uv.lock              # Reproducible lockfile
└── README.md
```

## Notebooks

Convención de nombres: `NN-seccion-descripcion.ipynb`. El prefijo numérico fija el orden de ejecución (Restart & Run All).

| Notebook | Sección del informe |
| --- | --- |
| `notebooks/00-intro-dataset.ipynb` | 3.1 Introducción |
| `notebooks/01-eda.ipynb` | 3.2 Análisis Exploratorio (EDA) |
| `notebooks/02-preprocessing.ipynb` | 3.3 Preprocesamiento |
| `notebooks/03-train-baseline-xgboost.ipynb` | 3.4 Entrenamiento — Modelo de ML clásico |
| `notebooks/04-train-nn-pytorch.ipynb` | 3.4 Entrenamiento — Red Neuronal |
| `notebooks/05-evaluation-comparison.ipynb` | 3.5 Evaluación y comparación |
| `notebooks/06-conclusions.ipynb` | 3.6 Conclusiones |

Para ejecutar todos en orden:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

## Capas de datos

| Carpeta | Contenido | Lectores | Escritores |
| --- | --- | --- | --- |
| `data/raw/` | Datos originales, sin tocar (read-only) | 00, 01, 02 | (descarga manual) |
| `data/interim/` | Resultados intermedios entre notebooks | según corresponda | 02 |
| `data/processed/` | Splits y features listos para modelar | 03, 04, 05 | 02 |

El contenido de `data/` no se versiona; solo se commitean los `.gitkeep`.

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — package & project manager

## Setup

El proyecto usa [`uv`](https://docs.astral.sh/uv/) para manejar dependencias y entorno virtual.

```bash
uv sync                       # crea .venv e instala las dependencias del lockfile
uv add <paquete>              # agregar una nueva dependencia
uv run jupyter lab            # ejecutar comandos dentro del entorno
uv run python src/script.py
```

## Stack sugerido

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, ydata-profiling
- XGBoost / LightGBM (modelo clásico)
- PyTorch (red neuronal, con soporte de GPU Apple Silicon / MPS)
- Jupyter / JupyterLab
