import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import preprocess_data


def get_feature_names(preprocessor):
    num_features = preprocessor.transformers_[0][2]
    cat_pipeline = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    onehot = cat_pipeline.named_steps['onehot']
    cat_feature_names = onehot.get_feature_names_out(cat_features)
    return list(num_features) + list(cat_feature_names)


def run_shap_analysis():
    # Paths
    data_path = "data/raw/noisy_manufacturing_quality_dataset.csv"
    model_path = "models/best_model.pkl"
    preprocessor_path = "models/preprocessor_pipeline.pkl"
    shap_summary_plot = "reports/shap_summary_plot.png"
    shap_force_plot_html = "reports/shap_force_plot.html"

    # Load data and model
    print("📦 Loading data and artifacts...")
    df = pd.read_csv(data_path, parse_dates=["start_time", "end_time"])
    preprocessor = joblib.load(preprocessor_path)
    model_obj = joblib.load(model_path)
    model = model_obj['model'] if isinstance(model_obj, dict) and "model" in model_obj else model_obj

    # Feature engineering
    df['defect_rate'] = df['defect_count'] / df['batch_size']
    df['processing_speed'] = df['batch_size'] / df['processing_time_min']
    features = df.drop(columns=['batch_id', 'start_time', 'end_time', 'quality_event'])

    # Transform features
    X_transformed = preprocessor.transform(features)
    feature_names = get_feature_names(preprocessor)
    X_matrix = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    X_df = pd.DataFrame(X_matrix, columns=feature_names)

    # Run SHAP
    print("🔍 Running SHAP...")
    explainer = shap.Explainer(model, X_matrix)
    shap_values = explainer(X_matrix)

    print("✅ SHAP computed.")
    print(f"SHAP shape: {shap_values.shape}")
    print(f"X_df shape: {X_df.shape}")

    # Extract class 1 SHAP values: shape = (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        print("ℹ️ Detected 3D SHAP output. Using SHAP values for class 1 (positive class).")
        shap_plot_data = shap.Explanation(
            values=shap_values.values[:, :, 1],  # ✅ Correct: [samples, features, class 1]
            base_values=shap_values.base_values[:, 1],
            data=X_matrix,
            feature_names=feature_names
        )
    else:
        shap_plot_data = shap_values

    print("✅ SHAP data prepared for beeswarm.")
    print(f"  - shap_plot_data.values.shape: {shap_plot_data.values.shape}")
    print(f"  - feature count: {len(shap_plot_data.feature_names)}")

    # Final shape check
    assert shap_plot_data.values.shape[1] == X_df.shape[1], \
        f"Shape mismatch! SHAP features: {shap_plot_data.values.shape[1]}, DataFrame features: {X_df.shape[1]}"
    print("🧪 Post-fix SHAP shape check passed!")

    # Plot beeswarm
    print("📊 Creating beeswarm plot...")
    os.makedirs("reports", exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(shap_plot_data, max_display=10, show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(shap_summary_plot)
    print(f"✅ SHAP summary plot saved to: {shap_summary_plot}")

    # Force plot for one record
    print("⚙️ Creating force plot for first record...")
    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    if len(shap_values.shape) == 3:
        explanation = shap.Explanation(
        values=shap_values.values[0, :, 1],
        base_values=shap_values.base_values[0, 1],
        data=X_matrix[0],
        feature_names=feature_names
    )
    else:
        explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_matrix[0],
        feature_names=feature_names
    )

# Generate the JS + HTML version of force plot
    print("⚙️ Creating force plot for first record...")
    force_plot = shap.plots.force(explanation, matplotlib=False)

# Save HTML content
    if hasattr(force_plot, "html"):
        with open(shap_force_plot_html, "w") as f:
            f.write(force_plot.html())
        print(f"✅ SHAP force plot saved to: {shap_force_plot_html}")
    else:
        print("❌ Failed to generate SHAP force plot HTML. The object is not valid.")

    shap.plots.force(explanation, matplotlib=True)
    plt.savefig("reports/shap_force_plot.png", bbox_inches='tight')


if __name__ == "__main__":
    run_shap_analysis()
