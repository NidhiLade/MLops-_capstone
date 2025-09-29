# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import argparse
import os
import pandas as pd
from preprocess import load_and_split

def train(csv_path, model_out_dir="models", n_estimators=100, random_state=42):
    # load
    X_train, X_test, y_train, y_test = load_and_split(csv_path)

    # set a human-friendly experiment name
    mlflow.set_experiment("mloops-capstone")

    with mlflow.start_run():
        params = {"n_estimators": n_estimators, "random_state": random_state, "model": "RandomForest"}
        mlflow.log_params(params)

        # train (keep labels as-is so the saved model predicts original labels)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # accuracy (classification)
        acc = accuracy_score(y_test, preds)

        # to compute a numeric MSE we map labels -> ints (only for the metric)
        labels = list(pd.concat([y_train, y_test]).unique())
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        y_test_num = [label_to_idx[y] for y in y_test]
        preds_num = [label_to_idx[p] for p in preds]
        mse = mean_squared_error(y_test_num, preds_num)

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("mse", float(mse))

        # save model artifact
        os.makedirs(model_out_dir, exist_ok=True)
        model_path = os.path.join(model_out_dir, "rf_model.pkl")
        joblib.dump(model, model_path)

        # log artifacts to MLflow
        mlflow.log_artifact(model_path, artifact_path="model")
        # also log as an MLflow model (sklearn format)
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        print(f"Trained model — accuracy={acc:.4f}, mse={mse:.4f}")
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/dataset.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train(args.csv, n_estimators=args.n_estimators)