# Diabetes Readmission Prediction with Plots

A machine learning project built with Python, scikit-learn, and seaborn to predict diabetes outcomes and visualize performance metrics using a Random Forest Classifier.

## 🚀 Features
- **Data Preprocessing**: Standardizes features using `StandardScaler` for optimal model performance.
- **Machine Learning Model**: Utilizes an ensemble `RandomForestClassifier` to classify diabetes risk.
- **Comprehensive Evaluation**: Generates Classification Reports and calculates the ROC-AUC score.
- **Advanced Visualizations**: Plots a 3-panel Matplotlib/Seaborn figure showing:
  1. Confusion Matrix
  2. Receiver Operating Characteristic (ROC) Curve with AUC
  3. Top 5 Feature Importances

---

## 📊 Dataset
The model uses the classic Pima Indians Diabetes Dataset hosted via [GitHub Datasets](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv). 

**Features included:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## 🛠️ Requirements & Installation

Make sure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
