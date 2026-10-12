# Importar librerías
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

# Separación por tipo de variable global
NUM_FEATURES = ["age", "study_hours_per_day", "social_media_hours", "netflix_hours",
                "attendance_percentage", "sleep_hours", "exercise_frequency",
                "mental_health_rating"]
ORD_FEATURES = ["diet_quality", "internet_quality", "parental_education_level"]
NOM_FEATURES = ["gender", "part_time_job", "extracurricular_participation"]

# Definifición de argumentos para el modelo
def arguments_parser():
    # Crear objeto parser
    parser = argparse.ArgumentParser(description="Entrenar Ridge para predecir exam_score")

    # Definir argumentos con su tipo de dato, valor por default, y su documentación de ayuda.
    parser.add_argument("--data", type = Path, default = None, help = "Ruta opcional al archivo CSV")
    parser.add_argument("--alpha", type = float, default = 1.0, help="Intensidad de regularización L2 (> 0)")
    parser.add_argument("--tol", type = float, default = 1e-4, help="Tolerancia del optimizador (> 0)")
    parser.add_argument("--max_iter", type = int, default = 500, help="Máximo de iteraciones (> 0)")
    parser.add_argument("--seed", type = int, default = 1234, help="Semilla aleatoria")

    return parser.parse_args()


# Cargar dataser
def load_data(dataset_path):
    data = pd.read_csv(dataset_path, index_col="student_id")
    X = data.drop("exam_score", axis = 1)
    y = data["exam_score"]
    return X, y

# Creación de pipeline
def preprocess_data(alpha, tol, max_iter, seed):
    ord_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),("ordinal", OrdinalEncoder(
            categories=[
                ["Poor", "Fair", "Good"],              # diet_quality
                ["Poor", "Average", "Good"],           # internet_quality
                ["High School", "Bachelor", "Master"]  # parental_education_level
        ]
        ))
        ])
    nom_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    num_transformer = Pipeline(steps=[
        ("scaler", MinMaxScaler())
        ])
    # Integración de los flujos de preprocesamiento según el tipo de variable 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_FEATURES),
            ("ord", ord_transformer, ORD_FEATURES),
            ("nom", nom_transformer, NOM_FEATURES)
            ]
            )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha, tol=tol, max_iter=max_iter,
                            solver="sag", random_state=seed)),
    ])

def metrics(model, X_train, y_train):
    # Cálculo de la predicción
    y_train_pred = model.predict(X_train)
    # Cálculo de las métricas
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)

    return {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
        }


def main():
    # Declarar el objeto argparser 
    args = arguments_parser()
    # Indicar Error por valores invalidos en hiperparámetros
    if args.alpha <= 0 or args.tol <= 0 or args.max_iter <= 0:
        raise ValueError("alpha, tol y max_iter deben ser mayores que cero")
    # Definir ruta del proyecto
    project_root = Path(__file__).resolve().parents[1]
    # Definir ruta de dataset
    dataset_path = args.data or project_root / "data" / "student_habits_performance.csv"
    # Recibir conjuto de caracteríticas y variable objetivo
    X, y = load_data(dataset_path)
    # Aplicar método de separación de prueba y entrenamieto
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state = args.seed
        )
    # Recibir modelo preprocesado y definido
    model = preprocess_data(args.alpha, args.tol, args.max_iter, args.seed)
    # Entrenar el modelo
    # Todas las corridas se guardan en el mismo directorio del proyecto.
    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # Declaración del url para el servidor de mlflow
    mlflow.set_tracking_uri(uri = "http://127.0.0.1:5000")
    #mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("Student_Performance_Ridge")
    with mlflow.start_run():
        for name, value in {
            "alpha": args.alpha, "tol": args.tol,
            "max_iter": args.max_iter, "seed": args.seed,
            "solver": "sag", "test_size": 0.2,
        }.items():
            mlflow.log_param(name, value)
        model.fit(X_train, y_train)
        # Calcular métricas del conjuto de entrenamiento y del conjunto de prueba
        for name, features, target in [("train", X_train, y_train), ("test", X_test, y_test)]:
            metrics_results = metrics(model, features, target)
            for metric_name, value in metrics_results.items():
                mlflow.log_metric(f"{name}_{metric_name}", float(value))
            print(f"{name.upper()}: MAE={metrics_results['mae']:.3f} | "
                  f"RMSE={metrics_results['rmse']:.3f} | R²={metrics_results['r2']:.3f}")
        predictions = model.predict(X_test)
        # Creacion de gráfica en artefactos
        with TemporaryDirectory() as tmp:
                    figure_path = Path(tmp) / "reales_vs_predichos.png"
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, predictions, alpha=0.6)
                    limits = [min(y_test.min(), predictions.min()),
                              max(y_test.max(), predictions.max())]
                    ax.plot(limits, limits, "--", color="gray")
                    ax.set(xlabel="Calificación real", ylabel="Calificación predicha",
                           title="Ridge: valores reales vs. predichos")
                    fig.tight_layout()
                    fig.savefig(figure_path, dpi=150)
                    plt.close(fig)
                    mlflow.log_artifact(str(figure_path), artifact_path="graficas")
        mlflow.sklearn.log_model(sk_model=model, name="modelo", serialization_format="cloudpickle")
    
if __name__ == "__main__":
    main()