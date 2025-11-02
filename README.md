# <p align="center">🏦 Python: Credit Risk Analysis and Anomaly Detection 🔎<p/>
<br>Created by Nattawut Boonnoon<br/>
LinkedIn: www.linkedin.com/in/nattawut-bn
<br>Email: nattawut.boonnoon@hotmail.com<br/>
Phone: (+66) 92 271 6680

This is my self-directed machine learning project for beginner-to-intermediate learners. It builds a credit risk analysis and fraud detection pipeline that uses only legitimate financial factors for risk scoring and applies unsupervised anomaly detection to flag potential fraud. The system generates an interactive Plotly dashboard (HTML) and a static PNG visualization for presentation.

***Objective***
-
1. Demonstrate Python Proficientcy: Implement data processing, analysis, and model prototyping using key Python libraries.
2. Explore Machine Learning Models: Develop and evaluate predictive models to assess credit risk and detect fraudulent activities.
3. Address Real-World Banking Challenges: Practices to improve credit assessment and strengthen fraud prevention in financial systems.


***Interactive Dashboards***
-
Potential fraud detection insights from the ML training models: <p><p/>
![Fraud Detection Summary Demo](Screenshot_Dashboard.png)

<p><p/>

Credit Risk Analysis results: <p><p/>
![Fraud Detection Summary Demo2](Screenshot_Dashboard2.png)

***Key features***
-
1. Robust data loading and validation
2. Feature engineering from legitimate financial attributes only
3. RandomForest risk model with optional hyperparameter search
4. IsolationForest-based anomaly detection with robust contamination estimation
5. Exports: interactive HTML dashboard and static PNG summary
6. Audit-ready logging and reproducible behavior (random_state)

# <p align="center">👩🏻‍💻 Run the Code ⚙️<p/>
1. Make sure you have Python 3.11 or newer installed on your system.
2. (Optional but recommended) Create a virtual environment:

- Run: python -m venv venv

- Activate it:

  - On Windows: venv\Scripts\activate
  - On Mac/Linux: source venv/bin/activate

3. Install all required packages by running: pip install -r requirements.txt
4. Run the file 'nattawut_credit_risk_refactored.py'.
5. After it runs, check the console output for summary results.
6. Open the folder credit_risk_outputs. You will find:

- dashboard_xxxxx.html (interactive dashboard, open in a web browser)

- summary_metrics.txt (summary_report)

- credit_risk_cover.png (cover visualization image)

7. If you want static images from Plotly charts, make sure kaleido is correctly installed.
8. Save results: click the camera icon or take a screenshot of the whole page (use Print Screen on Windows or Cmd+Shift+3 on macOS).

***Tips:***
-
- Check your data: ensure your CSV has columns like LoanAmount, Risk, and others listed in the README. Fix missing ones before uploading.
- Internet: needed the first time to install tools, but not to run afterward.
- Problems: look at credit_risk.log in your folder for clues, or try running again.

# <p align="center">⭐ Important Notices 📊<p/>
1. Data Hygiene: Validate input columns and types. Ensure LoanAmount > 0. Clean inconsistent categorical labels. Unknown categories are flagged by design.
2. Fair Lending & Compliance: Only financial attributes are used. Adhere to local regulations and keep transformation logic auditable.
3. Model & Fraud Limitations: Random Forest and Isolation Forest provide ~70–80% accuracy on real-world data. Still not fully 100%. Fraud detection flags anomalies but may produce false positives/negatives. Human judgment is required.
4. Performance & Reproducibility: For large datasets, scale resources as needed. Default random_state ensures reproducibility. Check credit_risk.log for warnings.
5. Visualization & Export: Plotly dashboards are interactive; static PNGs require kaleido or fallback to matplotlib/PIL.
6. Security & Privacy: Do not include sensitive personal data in public repos.
7. Project Scale: Mainly Educational, research, and demonstration purposes. Not for high-end financial or advanced credit decisions.
