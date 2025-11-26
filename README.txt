
# Credit Card Fraud Detection - Quick Start

## Files
- `main_fraud_detection.py`: main script
- `requirements.txt`: Python dependencies
- Place `creditcard.csv` (Kaggle dataset) next to the script.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main_fraud_detection.py --data creditcard.csv
```
Figures and the saved model will appear in the `outputs/` folder.

## Notes
- The script uses SMOTE inside an imblearn `Pipeline` to oversample *only* the training folds/sets.
- Two models are trained: RandomForest and, if installed, XGBoost. The best is chosen by validation PR-AUC.
- The final prediction threshold is tuned with F2-score to **reduce false negatives** (higher recall).
- Saved artifact: `outputs/<ModelName>_pipeline.joblib` containing both the pipeline and the tuned threshold.
