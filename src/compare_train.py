import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def eval_and_log(model_name, model, X_train, X_test, y_train, y_test, params=None):
    params = params or {}

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name}: RMSE={rmse:.4f}, R2={r2:.4f}")


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("diabetes-regression-baseline")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    eval_and_log(
        "linear_regression",
        LinearRegression(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    rf_params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}
    eval_and_log(
        "random_forest",
        RandomForestRegressor(**rf_params),
        X_train,
        X_test,
        y_train,
        y_test,
        rf_params,
    )

    gb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,"random_state": 42}
    eval_and_log(
        "gradient_boosting",
        GradientBoostingRegressor(**gb_params),
        X_train,
        X_test,
        y_train,
        y_test,
        gb_params,
    )


if __name__ == "__main__":
    main()