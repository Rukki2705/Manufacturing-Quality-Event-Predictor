import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


def preprocess_data(input_path: str, preprocessor_save_path: str = None):
    # Load the raw data
    df = pd.read_csv(input_path, parse_dates=["start_time", "end_time"])

    # Feature Engineering
    df['defect_rate'] = df['defect_count'] / df['batch_size']
    df['processing_speed'] = df['batch_size'] / df['processing_time_min']

    # Drop unused columns for modeling
    df_model = df.drop(columns=['batch_id', 'start_time', 'end_time'])

    # Define target and features
    target = 'quality_event'
    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Define feature types
    categorical_features = ['product_type', 'machine_id', 'operator_id', 'shift']
    numerical_features = [
        'batch_size', 'processing_time_min', 'inspection_duration_min',
        'defect_count', 'defect_rate', 'processing_speed'
    ]

    # Pipeline for numeric columns
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical columns
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine into a full preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Optionally save the preprocessor for reuse
    if preprocessor_save_path:
        os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_save_path)

    return X_processed, y


if __name__ == "__main__":
    input_path = "data/raw/noisy_manufacturing_quality_dataset.csv"
    save_path = "models/preprocessor_pipeline.pkl"

    print("[INFO] Preprocessing started...")
    X, y = preprocess_data(input_path, preprocessor_save_path=save_path)
    print(f"[INFO] Preprocessing complete. Transformed shape: {X.shape}")
    print(f"[INFO] Preprocessor saved to: {save_path}")

