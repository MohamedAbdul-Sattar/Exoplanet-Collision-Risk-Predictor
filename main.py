from pathlib import Path
from src.preprocess import preprocess
from src.train import train
from src.predict import test_model

if __name__ == "__main__":
    # Paths (use forward slashes via Path to avoid \p warnings)
    PROC = Path("data") / "processed"
    TRAIN_CSV = PROC / "koi_train.csv"
    TEST_CSV  = PROC / "koi_test.csv"
    MODEL_PKL = PROC / "koi_trained.pkl"
    TARGET = "koi_disposition"

    # 1) Preprocess (creates processed/koi_train.csv & koi_test.csv)
    preprocess()

    # 2) Train (saves model to processed/koi_trained.pkl)
    model_path = train(str(TRAIN_CSV), TARGET, str(MODEL_PKL))
    print(f"✅ Model saved to: {model_path}")

    # 3) Test (MODEL FIRST, then TEST CSV, then target)
    test_model(str(MODEL_PKL), str(TEST_CSV), TARGET)
