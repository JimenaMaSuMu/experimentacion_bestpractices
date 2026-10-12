## Organización del proyecto

```text
├── LICENSE                  <- Licencia de código abierto del proyecto
├── Makefile                 <- Comandos auxiliares para automatizar tareas
├── README.md                <- Documentación principal del proyecto
│
├── data
│   └── student_habits_performance.csv
│                              <- Dataset para entrenamiento y evaluación
│
├── docs                     <- Documentación generada con MkDocs
│
├── models                   <- Directorio destinado a modelos entrenados
│
├── notebooks
│   └── TC5061_Notebook_Baseline_SuarezJimena.ipynb
│                              <- Notebook original del proyecto de ML
│
├── pyproject.toml           <- Configuración del proyecto y versión de Python
│
├── references               <- Materiales de referencia y documentación adicional
│
├── reports                  <- Reportes y resultados del análisis
│   └── figures              <- Figuras utilizadas en los reportes
│
├── requirements.txt         <- Dependencias necesarias para reproducir el entorno
│
├── setup.cfg                <- Configuración de herramientas de desarrollo
│
└── experimentacion_bestpractices
    ├── __init__.py           <- Define el directorio como módulo de Python
    └── train.py              <- Script principal de entrenamiento, evaluación
                                 y registro de experimentos con MLflow
```

## Descripción del proyecto

Este proyecto tiene como objetivo implementar buenas prácticas de MLOps mediante la transformación de un flujo de Machine Learning desarrollado originalmente en un Jupyter Notebook hacia un script de Python modular y ejecutable desde la terminal.

El proyecto utiliza un conjunto de datos relacionado con los hábitos de estudio y las características personales de estudiantes para predecir su rendimiento académico mediante un modelo de regresión Ridge.

El notebook original se conserva en el directorio `notebooks/`, mientras que el archivo `experimentacion_bestpractices/train.py` contiene la implementación del flujo de entrenamiento, evaluación y seguimiento de experimentos.

## Instalación y configuración

Se recomienda utilizar Python 3.12 y crear un entorno virtual para instalar las dependencias del proyecto.

Desde la carpeta raíz del repositorio, ejecutar:

```bash
python -m venv .venv
```

Activar el entorno virtual en Git Bash para Windows:

```bash
source .venv/Scripts/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución del proyecto

### 1. Iniciar MLflow

Desde la carpeta raíz del proyecto, ejecutar:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

La interfaz de MLflow estará disponible en:

`http://127.0.0.1:5000`

Es necesario mantener esta terminal abierta mientras se ejecutan los entrenamientos.

### 2. Ejecutar el script de entrenamiento

En una segunda terminal, con el entorno virtual activado y ubicada en la raíz del proyecto:

```bash
python -m experimentacion_bestpractices.train
```

Este comando ejecuta el entrenamiento utilizando los valores predeterminados de los hiperparámetros.

Para modificar la configuración del modelo:

```bash
python -m experimentacion_bestpractices.train --alpha 0.1 --tol 0.0001 --max_iter 500 --seed 1234
```

Los argumentos disponibles son:

| Argumento | Descripción | Valor predeterminado |
|---|---|---|
| `--alpha` | Intensidad de regularización del modelo Ridge | 1.0 |
| `--tol` | Tolerancia utilizada por el algoritmo de optimización | 0.0001 |
| `--max_iter` | Número máximo de iteraciones | 500 |
| `--seed` | Semilla aleatoria para reproducibilidad | 1234 |

Cada ejecución registra sus resultados en el experimento `Student_Performance_Ridge` de MLflow.

## Mejoras implementadas de MLOps

Se incorporaron las siguientes mejoras:

- **Modularización:** organización del código en funciones para cargar los datos, realizar el preprocesamiento, entrenar el modelo y evaluar sus resultados.
- **Configuración mediante argumentos:** implementación de `argparse` para modificar los hiperparámetros `alpha`, `tol`, `max_iter` y la semilla aleatoria sin cambiar el código fuente.
- **Seguimiento de experimentos:** integración de MLflow para registrar los parámetros, métricas de entrenamiento y prueba, gráficas y modelos entrenados.
- **Comparación de resultados:** ejecución de diferentes configuraciones de Ridge Regression dentro de un mismo experimento denominado `Student_Performance_Ridge`.
- **Reproducibilidad:** documentación de dependencias, control de la aleatoriedad y uso de Git y GitHub para el control de versiones.


## Experimentación y resultados

Se realizaron diferentes ejecuciones del modelo Ridge Regression modificando los hiperparámetros `alpha`, `tol` y `max_iter`.

Para evaluar el desempeño se utilizaron las siguientes métricas:

- **MAE (Error Absoluto Medio):** promedio de las diferencias absolutas entre los valores reales y las predicciones.
- **RMSE (Raíz del Error Cuadrático Medio):** mide la magnitud de los errores, penalizando en mayor medida los errores grandes.
- **R² (Coeficiente de determinación):** indica qué proporción de la variabilidad de la variable objetivo es explicada por el modelo.

### Resultados obtenidos

| Alpha | Tol | Max. iteraciones | MAE prueba | RMSE prueba | R² prueba |
|---|---|---|---|---|---|
| **0.1** | **0.0001** | **500** | **4.6754** | **5.6067** | **0.9000** |
| 0.1 | 0.00001 | 2000 | 4.6755 | 5.6071 | 0.9000 |
| 1.0 | 0.0001 | 500 | 4.7247 | 5.7058 | 0.8965 |
| 1.0 | 0.001 | 1000 | 4.7231 | 5.7028 | 0.8966 |
| 1.0 | 0.00001 | 2000 | 4.7243 | 5.7052 | 0.8965 |
| 10.0 | 0.0001 | 500 | 6.1543 | 7.4989 | 0.8212 |

La mejor configuración observada fue `alpha=0.1`, `tol=0.0001` y `max_iter=500`, obteniendo un MAE de 4.6754, RMSE de 5.6067 y R² de aproximadamente 0.9000 en el conjunto de prueba.

Los resultados muestran que incrementar la regularización mediante `alpha=10` produjo un aumento considerable del error, mientras que las variaciones de tolerancia y número máximo de iteraciones tuvieron un efecto menor dentro de las configuraciones evaluadas.