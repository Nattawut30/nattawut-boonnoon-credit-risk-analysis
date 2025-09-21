# Credit Risk Analysis & Fraud Detection Dashboard
This is my personal Python simple credit risk analysis and fraud detection pipeline. Uses only legitimate financial factors for risk scoring and unsupervised anomaly detection for potential fraud. Produces an interactive Plotly dashboard (HTML) and a static PNG cover image.

*(Example: Summary Report)* <p><p/>
![Credit Analysis Console Demo](Screenshot_Console.png) <p><p/>
*(HTML: Screenshot Dashboard from Plotly)* <p><p/>
![Fraud Detection Summary Demo](Screenshot_Dashboard.png) <p><p/>

***Key features***

1. Robust data loading and validation
2. Feature engineering from legitimate financial attributes only
3. RandomForest risk model with optional hyperparameter search
4. IsolationForest-based anomaly detection with robust contamination estimation
5. Exports: interactive HTML dashboard and static PNG summary
6. Audit-ready logging and reproducible behavior (random_state)

# Installation
***Prerequisites:***
- Python 3.9 or newer
- Recommended virtual environment

***Create virtual environment and install packages:***

python -m venv venv
- macOS / Linux ==> *source* venv/bin/activate
- Windows (PowerShell) ==> venv\Scripts\Activate.ps1

pip install --upgrade pip<p><p/>
pip install pandas numpy scikit-learn plotly matplotlib pillow joblib
- Optional (to enable Plotly -> PNG export) ==> pip install kaleido
- Optional (for Streamlit UI if desired) ==> pip install streamlit
<p><p/>
  
***Pro Tips:*** <p><p/>
- kaleido is optional but required if you want the interactive Plotly figure saved as a PNG. If not installed the script falls back to a matplotlib-based PNG summary.
- Use joblib to save and load models.

# How to Run the Code
-
-

# Important Notes
1. Data hygiene:
Validate input columns and types before running.
Ensure LoanAmount is numeric and > 0.
Clean inconsistent categorical labels (e.g., typos in CreditHistory) or the script will log warnings and treat them as missing. Mappings return NaN for unknown categories by design so you can detect data problems rather than silently accepting incorrect encodings.

2. Fair lending and compliance:
This pipeline uses only financial attributes for scoring and no protected-class variables. Still adhere to local fair-lending regulations. Keep transformation logic auditable.
Maintain versioned snapshots of training data, model artifacts, and transformation code for audits.

3. Model tuning and production:
RandomizedSearchCV is used to limit runtime. For production, run exhaustive tuning on a curated compute node and store the final best_estimator_.
Monitor model drift. Retrain on new labeled data periodically.

4. Fraud detection limitations:
IsolationForest is unsupervised and flags statistical anomalies. It will produce false positives and false negatives.
Use rule-based heuristics and human review pipelines to validate flagged cases.

5. Performance and scaling:
For large datasets, increase compute resources and consider using incremental training or distributed frameworks.5. 
Grid/random search is computationally intensive. Use --quick for exploration.

6. Reproducibility & logs:
Default random_state is set for reproducibility. Change as needed.
Check credit_risk.log for warnings about unrecognized categories, missing columns, or imputation decisions.

7. Visualization & export:
If kaleido is installed, the script will try to export a PNG from Plotly. If not installed the script generates a static PNG using matplotlib and PIL.
The interactive dashboard HTML can be opened with any modern browser.

8. Security & privacy:
Do not place sensitive personal data (PII) in public repositories.
Mask or redact PII before sharing outputs.


# Credits
Built By Nattawut Boonnoon <p><p/>
Created On Jan 30, 2025 <p><p/>
**Latest Updated: Sept. 21, 2025 > Repo_404_Error, Status: Fixed* <p><p/>
LinkedIn: www.linkedin.com/in/nattawut-bn
