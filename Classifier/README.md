# Document Layout Analysis: Classification Models

This project contains machine learning scripts designed to classify document elements (e.g., headers, footers, body text) based on their visual and textual features. It includes implementations for Naive Bayes, Random Forest, and XGBoost classifiers.

## 1. Prerequisites

Ensure you have Python installed (3.8+ recommended). Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost joblib

Project_Root/
ÃÄÄ Assets/
³   ÀÄÄ output/
³       ÀÄÄ combine/
³           ÀÄÄ dataset.csv           # Input data source
ÃÄÄ Classifier/
³   ÃÄÄ NaiveBayes/
³   ³   ÀÄÄ model_nb.py               # Naive Bayes script
³   ÃÄÄ RandomForest/
³   ³   ÀÄÄ model_rf.py               # Random Forest script
³   ÀÄÄ XGBoost/
³       ÀÄÄ model_xgb.py              # XGBoost script
ÃÄÄ Results/
³   ÃÄÄ NaiveBayes/                   # Output folder for Naive Bayes
³   ÃÄÄ RandomForest/                 # Output folder for Random Forest
³   ÀÄÄ XGBoost/                      # Output folder for XGBoost
ÀÄÄ README.md

python Classifier\NaiveBayes\model_nb.py -i Assets\output\combine\dataset.csv -o Results\NaiveBayes

python Classifier\RandomForest\model_rf.py -i Assets\output\combine\dataset.csv -o Results\RandomForest

python Classifier\XGBoost\model_xgb.py -i Assets\output\combine\dataset.csv -o Results\XGBoost
