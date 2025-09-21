# Credit Risk Analysis & Fraud Detection Dashboard
Welcome to my personal Python beginner-friendly credit risk analysis and fraud detection pipeline. Uses only legitimate financial factors for risk scoring and unsupervised anomaly detection for potential fraud. Produces an interactive Plotly dashboard (HTML) and a static PNG cover image.

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

# How to Run the Code
1. Make sure you have Python 3.11 or newer installed on your system.

2. Open a terminal or command prompt.

3. Go to the project folder where 'Nattawut_Credit.py' and 'requirements.txt' are located.

4. (Optional but recommended) Create a virtual environment:

- Run: python -m venv venv

- Activate it:

  - On Windows: venv\Scripts\activate
  - On Mac/Linux: source venv/bin/activate

5. Install all required packages by running: pip install -r requirements.txt

6. Run the program with: python 'Nattawut_Credit.py'

7. After it runs, check the console output for summary results.

8. Open the folder credit_risk_outputs. You will find:

- dashboard_xxxxx.html (interactive dashboard, open in a web browser)

- summary_metrics.csv (summary data)

- credit_risk_cover.png (cover visualization image)

9. If you want static images from Plotly charts, make sure kaleido is correctly installed. <p><p/>
10. Save results: click the camera icon on any picture to save it, or take a screenshot of the whole page (use Print Screen on Windows or Cmd+Shift+3 on Mac).

***Pro Tips:***
- Check your data: ensure your CSV has columns like LoanAmount, Risk, and others listed in the README. Fix missing ones before uploading.
- Internet: needed the first time to install tools, but not to run afterward.
- Problems: look at credit_risk.log in your folder for clues, or try running again.

# Important Notices
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
**Latest Updated: Sept. 21, 2025 > Repo_404_Error = Status: Fixed* <p><p/>
LinkedIn: www.linkedin.com/in/nattawut-bn
