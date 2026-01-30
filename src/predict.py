import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def test_model(model_path, test_file_path, target_column):
    """
    Loads a trained model and evaluates it on a test dataset.

    Parameters:
    - model_path: str, path to the trained model (.pkl)
    - test_file_path: str, path to the test dataset CSV
    - target_column: str, the name of the target column in the test data
    """

    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading test data...")
    test_data = pd.read_csv(test_file_path)

    # Separate features and labels
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")

