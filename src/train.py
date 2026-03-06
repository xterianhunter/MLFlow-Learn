import os
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # 1) Tell client where tracking server is
    print("Opening MLFLOW")
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Load dataset
    print("Loading dataset")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": 42,
    }

    # 3) Start one tracked run
    print("Training")
    with mlflow.start_run(run_name="rf-baseline"):
        # 4) Log hyperparameters
        print("Metrics")
        mlflow.log_params(params)

        print("RF")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        print("predict")
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        # 5) Log metrics
        print("Logging Metrics")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # 6) Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 7) Log a plot artifact
        os.makedirs("artifacts", exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, preds, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plot_path = "artifacts/actual_vs_pred.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(plot_path)

        # Optional: log metrics JSON as artifact
        metrics_path = "artifacts/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"rmse": rmse, "r2": r2}, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print(f"Run complete. RMSE={rmse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    main()