# 🧪 Manufacturing Quality Event Predictor

A complete machine learning pipeline to simulate, predict, and explain manufacturing batch-level quality risks using SHAP and LLMs. Built with modular Python scripts and a Streamlit-based explainability dashboard.

---

## 🔍 Project Highlights

- 🏗️ Modular data generation, preprocessing, and model training pipeline
- ✅ SHAP-based model explainability for both global and local interpretations
- 🔧 What-if simulator to modify inputs and visualize prediction changes
- 🧠 LLaMA3-powered summary generation via Groq API
- 📈 Streamlit dashboard for real-time QA exploration

---

## 🗂️ Project Structure

```
├── app/
│ └── streamlit_app.py # Streamlit UI for live prediction & SHAP
├── data/
│ └── raw/noisy_manufacturing_quality_dataset.csv
├── models/
│ ├── best_model.pkl # Trained model (RandomForest/XGBoost)
│ └── preprocessor_pipeline.pkl # Fitted ColumnTransformer
├── reports/
│ ├── shap_summary_plot.png # SHAP beeswarm plot
│ ├── shap_force_plot.html # SHAP force plot (HTML)
│ └── model_metrics.json # F1 scores, test reports
├── src/
│ ├── data_preparation.py # Synthetic dataset generator (Faker)
│ ├── feature_engineering.py # Preprocessing & transformation pipeline
│ ├── model_training.py # GridSearchCV, training, evaluation, export
│ └── shap_analysis.py # SHAP visualization and explanation generator
```
## 🚀 How to Run the Project

### 1️⃣ Clone the Repository & Install Dependencies
```
git clone https://github.com/Rukki2705/Manufacturing-Quality-Event-Predictor.git
cd quality-event-predictor
pip install -r requirements.txt
```
### 2️⃣ Generate Simulated Manufacturing Data
```
python src/data_preparation.py
```
### 3️⃣ Preprocess Data & Train ML Models
```
python src/feature_engineering.py
python src/model_training.py
```
### 4️⃣ Run SHAP Analysis (Optional)
```
python src/shap_analysis.py
```
### 5️⃣ Launch Streamlit QA Dashboard
```
streamlit run app/streamlit_app.py
```
---
## 🧠 Groq API (LLaMA3)
- To enable QA summary generation using LLaMA3:

- Enter your Groq API Key in the Streamlit sidebar at runtime.

---
## 🧰 Tech Stack
- Python, pandas, scikit-learn, XGBoost

- SHAP, matplotlib

- Faker (for synthetic data)

- Streamlit for UI

- Groq API + LLaMA3 for summary generation

---
## 👨‍💻 Author

**Hrushikesh Attarde**  
[LinkedIn](https://www.linkedin.com/in/hrushikesh-attarde) · [GitHub](https://github.com/Rukki2705)
