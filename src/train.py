import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(train_file_path: str, target_column: str, model_output_path: str) -> str:
    """
    Trains a Random Forest model on a CSV dataset and returns the saved model path.
    Assumes the CSV is already preprocessed (all features numeric) and split.
    """
    # 1) Load
    data = pd.read_csv(train_file_path)
    if data.empty:
        raise ValueError(f"Training CSV is empty: {train_file_path}")
    if target_column not in data.columns:
        raise KeyError(f"Target '{target_column}' not found in {train_file_path}")

    # 2) Split features/target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 3) Validate features are numeric (preprocess should have handled this)
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise TypeError(
            "Non-numeric columns found. Ensure preprocessing encodes everything.\n"
            f"Non-numeric: {non_numeric}"
        )

    # 4) Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # 5) Save & return path
    joblib.dump(model, model_output_path)
    print(f"Model trained on {len(data)} samples and saved to {model_output_path}")
    return model_output_path
