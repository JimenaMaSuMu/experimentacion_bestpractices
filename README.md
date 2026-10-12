## 1. Descripción del proyecto

Este proyecto tiene como objetivo implementar buenas prácticas de MLOps mediante la migración de un flujo de Machine Learning desarrollado originalmente en un Jupyter Notebook hacia un script de Python ejecutable desde la terminal.

El notebook original, ubicado en el directorio `notebooks/`, contiene el desarrollo inicial del modelo de predicción del rendimiento académico de estudiantes a partir de sus hábitos y características personales y su exploración de datos.

A partir de este trabajo se desarrolló `experimentacion_bestpractices/train.py`, que centraliza la carga y el preprocesamiento de datos, el entrenamiento de un modelo Ridge Regression y su evaluación. Además, incorpora `argparse` para configurar hiperparámetros y MLflow para registrar experimentos, métricas, modelos y artefactos.

Esta implementación permite ejecutar y comparar diferentes configuraciones del modelo sin modificar directamente el código fuente, mejorando la reproducibilidad, trazabilidad y organización del proceso de experimentación.

Archivos principales:

notebooks/: contiene el Jupyter Notebook original, utilizado como punto de partida para el desarrollo del proyecto.

experimentacion_bestpractices/train.py: script principal que implementa el flujo de entrenamiento, evaluación y registro de experimentos con MLflow.

data/student_habits_performance.csv: conjunto de datos utilizado para entrenar y evaluar el modelo.

requirements.txt: dependencias necesarias para ejecutar el proyecto.

reports/: directorio destinado a los resultados y evidencias del proyecto.

# Experimentacion_BestPractices

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Experimento para implementar las mejores practicas dentro de MLOps

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         experimentacion_bestpractices and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── experimentacion_bestpractices   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes experimentacion_bestpractices a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

