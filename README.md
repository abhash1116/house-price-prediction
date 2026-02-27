# Titanic Survival Prediction (Kaggle Competition)

Predict passenger survival on the Titanic using machine learning.

## Overview
- Competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Task: Binary classification
- Best validation accuracy: ~84% (Random Forest)
- Public leaderboard score: [your score, e.g., 0.782]

## Key Learnings / Techniques
- EDA & visualization (Seaborn)
- Feature engineering: Title extraction, FamilySize, Age bins
- Handling missing values & categoricals (one-hot encoding)
- Models: Logistic Regression → Random Forest → Comparison
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

## Files
- `Titanic_EDA_Modeling.ipynb` → Full notebook
- `train.csv` / `test.csv` → Data
- `submission.csv` → Sample submission
- `feature_importance.png` → Random Forest importances

## How to Run
1. Clone repo: `git clone https://github.com/yourusername/titanic-survival-prediction.git`
2. Install deps: `pip install -r requirements.txt`
3. Open `Titanic_EDA_Modeling.ipynb` in Jupyter/Colab/VS Code

## Results
- Cross-val accuracy: ~0.83
- Leaderboard: [link or screenshot]

## Future Improvements
- Try XGBoost / LightGBM
- Add SHAP explanations
- Ensemble models
