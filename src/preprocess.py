import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# Paths
def preprocess(): 
    RAW_PATH = 'data/raw/koi.csv'
    PROCESSED_DIR = 'data/processed'

    # Make sure the processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load the dataset
    df = pd.read_csv(RAW_PATH, comment = "#")

    # 2. Identify the target and features
    TARGET = 'koi_disposition'

    # Numeric features to use
    NUMERIC_FEATURES = [
        'koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
        'koi_tce_plnt_num', 'koi_steff', 'koi_slogg', 'koi_srad'
    ]

    # 3. Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # 4. Fill missing numeric feature values with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 5. Separate features and target
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]

    # 6. Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE' -> integers

    # 7. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 9. Save training and testing sets to CSV in processed folder
    train_df = pd.DataFrame(X_train, columns=NUMERIC_FEATURES)
    train_df[TARGET] = y_train
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'koi_train.csv'), index=False)

    test_df = pd.DataFrame(X_test, columns=NUMERIC_FEATURES)
    test_df[TARGET] = y_test
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'koi_test.csv'), index=False)

    print("Preprocessing complete!")
    print(f"Training data saved as '{os.path.join(PROCESSED_DIR, 'koi_train.csv')}'")
    print(f"Testing data saved as '{os.path.join(PROCESSED_DIR, 'koi_test.csv')}'")
