# House Prices Prediction (Kaggle Competition)

Predict residential home sale prices in Ames, Iowa using advanced regression techniques.

## Overview
- Competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Task: Regression (predict continuous SalePrice)
- Best validation RMSLE: ~0.13 (XGBoost with log-transformed target)
- Public leaderboard score:  0.132

## Key Learnings / Techniques
- Exploratory Data Analysis (EDA) & visualization (Seaborn, Matplotlib)
- Feature engineering: TotalSF, YrSoldRemod, interactions (e.g., OverallQual × GrLivArea), Age/Fare-like bins for LotArea/GrLivArea
- Handling missing values (group-by imputation, filling with 'None'/0)
- Target transformation: log1p(SalePrice) for skewed distribution
- Encoding: OrdinalEncoder for categoricals, one-hot where needed
- Models: XGBoost Regressor (main), comparisons with Ridge/Lasso/RandomForest
- Metrics: RMSLE (official), RMSE (log scale), R², MAE
- Cross-validation & early stopping to prevent overfitting
- Feature importance visualization (XGBoost built-in plot)

## Files
- `House_Prices` → Full notebook with preprocessing, modeling, and evaluation
- `train(2).csv` / `test(2).csv` → Original Kaggle data
- `submission.csv` → Sample/best submission file
- `data_description.txt` → Official variable descriptions (optional, but helpful)
 
requirements → Dependencies (pandas, numpy, xgboost, scikit-learn, etc.)

## How to Run
1. Clone the repo:
git clone https://github.com/yourusername/house-prices-xgboost.git
cd house-prices-xgboost
text2. Install dependencies:
pip install -r requirements.txt
text3. Open the notebook in Jupyter, VS Code, or Google Colab:
jupyter notebook House_Prices_EDA_Modeling.ipynb
textor upload to Colab and run all cells.

## Results
- Cross-validation RMSLE: ~0.1269 ± 0.0105 (5-fold)
- Approximate RMSE in dollars: ~$20k–25k (on validation)
- Leaderboard: 0.13282

## Future Improvements
- Advanced hyperparameter tuning (Optuna/Bayesian optimization)
- More feature engineering (polynomials, rare category grouping, external data if allowed)
- Model ensembling/stacking (XGBoost + LightGBM + Linear models)
- Outlier detection & removal (e.g., clip extreme GrLivArea/SalePrice)
- SHAP or LIME for interpretability
- Try neural networks (simple MLP) for comparison

Feel free to fork, star, or use this as a template for your own Kaggle projects!
