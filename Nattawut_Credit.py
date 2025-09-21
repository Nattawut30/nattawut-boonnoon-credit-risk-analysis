# Simple Credit Risk Analysis & Fraud detections Dashboard By Nattawut Boonnoon
# GitHub: @Nattawut30

#Update Sep 19, 2025: Repo_404_Error Fixed!

#!/usr/bin/env python3
"""
Credit Risk & Fraud Detection - Final Script
Fixed and hardened version. Produces:
 - Feature engineering (legitimate financial factors only)
 - Risk model training (RandomForest with optional RandomizedSearchCV)
 - Anomaly/fraud detection (IsolationForest with robust contamination estimation)
 - Interactive dashboard export (HTML) and static dashboard cover (PNG)
 - Text risk report
Usage:
    python credit_risk_dashboard.py --data path/to/data.csv --output-dir ./outputs --quick
"""

import os
import sys
import argparse
import logging
import datetime
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# Logging setup
LOG_FILE = "credit_risk.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CreditRisk")


class CreditRiskAnalyzer:
    """
    Credit risk & fraud analyzer.
    Only legitimate financial features are used.
    """

    def __init__(self, data_path: Optional[str] = None, random_state: int = 42):
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.risk_model = None
        self.fraud_model = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        if data_path:
            loaded = self.load_data(data_path)
            if not loaded:
                raise FileNotFoundError(f"Failed to load data: {data_path}")

    # -------------------------
    # Data loading & sanitation
    # -------------------------
    def load_data(self, data_path: str) -> bool:
        """Load CSV and apply robust missing-value handling and basic validation."""
        if not os.path.exists(data_path):
            logger.error("File not found: %s", data_path)
            return False
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                logger.error("Loaded file is empty: %s", data_path)
                return False
            # Basic required columns check (non-exhaustive)
            required = [
                'LoanAmount', 'LoanDuration', 'InstallmentPercent', 'Age',
                'EmploymentDuration', 'CreditHistory', 'ExistingSavings',
                'OwnsProperty', 'LoanPurpose', 'CurrentResidenceDuration',
                'ExistingCreditsCount', 'Risk'
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error("Missing required columns: %s", missing)
                # still set df so user can adapt; return False to indicate problem
                self.data = df
                return False

            # Convert numeric-like strings if possible
            for col in df.columns:
                if df[col].dtype == object:
                    # try to coerce numeric-like columns
                    try:
                        coerced = pd.to_numeric(df[col], errors='coerce')
                        if coerced.notna().sum() / len(coerced) > 0.6:
                            df[col] = coerced
                    except Exception:
                        pass

            # Remove clearly invalid rows
            if 'LoanAmount' in df.columns:
                df = df[df['LoanAmount'].notna() & (df['LoanAmount'] > 0)]

            # Fill missing values robustly
            for col in df.columns:
                if is_numeric_dtype(df[col]):
                    median = df[col].median()
                    df[col].fillna(median, inplace=True)
                else:
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col].fillna(mode.iloc[0], inplace=True)
                    else:
                        df[col].fillna("unknown", inplace=True)

            self.data = df.reset_index(drop=True)
            logger.info("Data loaded: %d rows, %d cols", len(self.data), len(self.data.columns))
            return True
        except Exception as exc:
            logger.exception("Exception while loading data: %s", exc)
            return False

    # -------------------------
    # Mapping helpers
    # -------------------------
    def _map_with_warn(self, mapping: dict, value, name: str):
        if pd.isna(value):
            return np.nan
        if value in mapping:
            return mapping[value]
        else:
            # prefer np.nan so unknown categories are handled explicitly later
            logger.warning("Unrecognized value for %s: %s", name, str(value))
            return np.nan

    def map_employment_duration(self, value):
        mapping = {
            'unemployed': 1,
            '<1 year': 2,
            '1<=X<4 years': 3,
            '4<=X<7 years': 4,
            '>=7 years': 5
        }
        return self._map_with_warn(mapping, value, "EmploymentDuration")

    def map_credit_history(self, value):
        mapping = {
            'no credits taken/all credits paid back duly': 5,
            'all credits at this bank paid back duly': 4,
            'existing credits paid back duly till now': 3,
            'delay in paying off in the past': 2,
            'critical account/other credits existing': 1
        }
        return self._map_with_warn(mapping, value, "CreditHistory")

    def map_savings(self, value):
        mapping = {
            'no known savings': 1,
            '<100 DM': 2,
            '100<=X<500 DM': 3,
            '500<=X<1000 DM': 4,
            '>=1000 DM': 5
        }
        return self._map_with_warn(mapping, value, "ExistingSavings")

    def map_property(self, value):
        mapping = {
            'no known property': 1,
            'car or other': 2,
            'building soc. savings agr./life insurance': 3,
            'real estate': 4
        }
        return self._map_with_warn(mapping, value, "OwnsProperty")

    def map_loan_purpose(self, value):
        mapping = {
            'education': 1,
            'domestic appliances': 2,
            'repairs': 2,
            'radio/television': 3,
            'new car': 3,
            'used car': 4,
            'furniture/equipment': 4,
            'business': 5,
            'vacation/others': 5
        }
        return self._map_with_warn(mapping, value, "LoanPurpose")

    # -------------------------
    # Feature engineering
    # -------------------------
    def create_legitimate_features(self):
        """Create features derived only from legitimate financial inputs."""
        if self.data is None:
            logger.error("No data to process.")
            return None
        df = self.data.copy()

        # Apply mapping functions with explicit NaN handling
        df['employment_stability_score'] = df['EmploymentDuration'].apply(self.map_employment_duration)
        df['credit_history_score'] = df['CreditHistory'].apply(self.map_credit_history)
        df['savings_score'] = df['ExistingSavings'].apply(self.map_savings)
        df['property_ownership_score'] = df['OwnsProperty'].apply(self.map_property)
        df['loan_purpose_risk'] = df['LoanPurpose'].apply(self.map_loan_purpose)

        # Numeric pass-throughs
        df['installment_burden'] = df.get('InstallmentPercent', 0).astype(float)
        df['residence_stability'] = df.get('CurrentResidenceDuration', 0).astype(float)
        df['existing_credits_burden'] = df.get('ExistingCreditsCount', 0).astype(float)
        df['LoanAmount'] = df['LoanAmount'].astype(float)
        df['LoanDuration'] = df['LoanDuration'].astype(float)
        df['Age'] = df['Age'].astype(float)

        # Optional Income-derived feature
        if 'Income' in df.columns:
            df['loan_to_income_ratio'] = df['LoanAmount'] / df['Income'].clip(lower=1)
        else:
            df['loan_to_income_ratio'] = np.nan
            logger.debug("Income column not present; loan_to_income_ratio set to NaN")

        # Fraud detection helpers
        df['age_loan_ratio'] = df['Age'] / df['LoanAmount'].clip(lower=1) * 1000
        df['duration_amount_ratio'] = df['LoanDuration'] / df['LoanAmount'].clip(lower=1) * 100

        # Composite scores (equal weights by design, can be tuned later)
        df['financial_stability_score'] = (
            df['employment_stability_score'].fillna(df['employment_stability_score'].median()) * 0.25 +
            df['savings_score'].fillna(df['savings_score'].median()) * 0.25 +
            df['property_ownership_score'].fillna(df['property_ownership_score'].median()) * 0.25 +
            df['credit_history_score'].fillna(df['credit_history_score'].median()) * 0.25
        )

        df['risk_score'] = (
            df['installment_burden'] * 0.3 +
            df['loan_purpose_risk'].fillna(df['loan_purpose_risk'].median()) * 0.2 +
            df['existing_credits_burden'] * 0.3 +
            (6 - df['financial_stability_score']) * 0.2
        )

        # target binary
        if 'Risk' in df.columns:
            df['risk_binary'] = (df['Risk'].astype(str).str.lower() == 'bad').astype(int)
        else:
            df['risk_binary'] = 0
            logger.warning("Risk column missing. risk_binary set to 0 for all rows.")

        # Final cleanup: ensure no infinite or extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        self.processed_data = df
        logger.info("Feature engineering completed. Processed rows: %d", len(df))
        return df

    # -------------------------
    # Fraud detection
    # -------------------------
    def detect_anomalies(self, contamination: Optional[float] = None, features: Optional[List[str]] = None):
        """
        Detect potential fraud using IsolationForest.
        If contamination is None estimate using row-level IQR outlier detection.
        """
        if self.processed_data is None:
            logger.error("Processed data missing. Run create_legitimate_features() first.")
            return None

        if features is None:
            features = [
                'LoanAmount', 'LoanDuration', 'InstallmentPercent',
                'Age', 'financial_stability_score', 'risk_score',
                'age_loan_ratio', 'duration_amount_ratio'
            ]
        X_fraud = self.processed_data[features].copy()

        # Fill numeric na with median
        for col in X_fraud.columns:
            if is_numeric_dtype(X_fraud[col]):
                X_fraud[col].fillna(X_fraud[col].median(), inplace=True)
            else:
                # Should not happen for chosen features, but safe fallback
                X_fraud[col] = pd.to_numeric(X_fraud[col], errors='coerce').fillna(0)

        # Estimate contamination as fraction of rows with any IQR-based outlier
        if contamination is None:
            q1 = X_fraud.quantile(0.25)
            q3 = X_fraud.quantile(0.75)
            iqr = q3 - q1
            mask = ((X_fraud < (q1 - 1.5 * iqr)) | (X_fraud > (q3 + 1.5 * iqr))).any(axis=1)
            outlier_rows = int(mask.sum())
            contamination = float(min(max(outlier_rows / max(len(X_fraud), 1), 0.01), 0.3))
            logger.info("Estimated contamination based on IQR outlier rows: %d / %d -> %.3f",
                        outlier_rows, len(X_fraud), contamination)

        # Fit IsolationForest
        self.fraud_model = IsolationForest(contamination=contamination, random_state=self.random_state)
        try:
            preds = self.fraud_model.fit_predict(X_fraud)
        except Exception:
            # Fallback: convert to numpy array if feature names cause issue
            preds = self.fraud_model.fit_predict(X_fraud.values)
        self.processed_data['potential_fraud'] = (preds == -1).astype(int)
        fraud_count = int(self.processed_data['potential_fraud'].sum())
        logger.info("Detected %d potential fraud cases (%.2f%%)", fraud_count,
                    fraud_count / len(self.processed_data) * 100)
        return preds

    # -------------------------
    # Model training
    # -------------------------
    def train_risk_model(self, quick: bool = False, random_search_iters: int = 20):
        """
        Train a RandomForest classifier to predict risk_binary.
        quick=True uses a single RandomForest (no hyperparameter search).
        """
        if self.processed_data is None:
            logger.error("Processed data missing. Run create_legitimate_features() first.")
            return None

        risk_features = [
            'employment_stability_score', 'credit_history_score', 'savings_score',
            'property_ownership_score', 'installment_burden', 'loan_purpose_risk',
            'existing_credits_burden', 'residence_stability', 'LoanAmount', 'LoanDuration'
        ]
        X = self.processed_data[risk_features].copy()
        for col in X.columns:
            if is_numeric_dtype(X[col]):
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        y = self.processed_data['risk_binary']
        if y.nunique() < 2:
            logger.warning("Not enough classes in target to train. Aborting training.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        if quick:
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=self.random_state)
            clf.fit(X_train, y_train)
            self.risk_model = clf
            logger.info("Quick RandomForest trained (no hyperparameter search).")
        else:
            # Randomized search to limit runtime while exploring hyperparameters
            param_dist = {
                'n_estimators': [50, 100, 200, 400],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2', None]
            }
            base = RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
            search = RandomizedSearchCV(
                base, param_distributions=param_dist, n_iter=min(random_search_iters, 30),
                scoring='f1_weighted', cv=3, n_jobs=-1, random_state=self.random_state, verbose=0
            )
            search.fit(X_train, y_train)
            self.risk_model = search.best_estimator_
            logger.info("RandomizedSearchCV done. Best params: %s", getattr(search, 'best_params_', {}))

        # Evaluate with correct F1 metric
        y_train_pred = self.risk_model.predict(X_train)
        y_test_pred = self.risk_model.predict(X_test)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        logger.info("Training F1 (weighted): %.3f", train_f1)
        logger.info("Testing F1 (weighted):  %.3f", test_f1)
        logger.info("Classification report (test):\n%s", classification_report(y_test, y_test_pred, digits=3))

        # Store feature importance in processed_data for visualization later
        if hasattr(self.risk_model, "feature_importances_"):
            fi = pd.Series(self.risk_model.feature_importances_, index=risk_features).sort_values(ascending=False)
            self.feature_importance_ = fi
        else:
            self.feature_importance_ = pd.Series(dtype=float)

        return self.risk_model

    # -------------------------
    # Visualization exports
    # -------------------------
    def create_dashboard_visualizations(self, output_dir: str = ".", html_name: str = "dashboard.html",
                                        png_name: str = "dashboard_cover.png", try_kaleido: bool = True):
        """
        Create Plotly interactive dashboard (HTML) and a static PNG cover.
        If kaleido is available and try_kaleido True the script will attempt to write a PNG via Plotly.
        Otherwise it will produce a matplotlib-based static cover PNG as fallback.
        """
        if self.processed_data is None:
            logger.error("No processed data. Cannot create visualizations.")
            return None

        df = self.processed_data.copy()
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, html_name)
        png_path = os.path.join(output_dir, png_name)

        # Build interactive Plotly figure (multi-panel)
        try:
            specs = [[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                     [{'type': 'xy'}, {'type': 'xy'}, {'type': 'domain'}]]
            fig = make_subplots(rows=2, cols=3, specs=specs,
                                subplot_titles=[
                                    "Bad Rate by Financial Stability", "Loan Amount Distribution by Risk",
                                    "Risk vs Installment Burden", "Credit History vs Bad Rate",
                                    "Feature Importance (Top)", "Fraud Flags"
                                ])
            # 1. Bad rate by financial stability
            stab = df.groupby('financial_stability_score')['risk_binary'].mean().reset_index()
            fig.add_trace(go.Bar(x=stab['financial_stability_score'], y=stab['risk_binary']*100,
                                 name='Bad Rate'), row=1, col=1)

            # 2. Loan amount distribution
            fig.add_trace(go.Histogram(x=df[df['risk_binary'] == 0]['LoanAmount'], nbinsx=30, name='Good'), row=1, col=2)
            fig.add_trace(go.Histogram(x=df[df['risk_binary'] == 1]['LoanAmount'], nbinsx=30, name='Bad'), row=1, col=2)

            # 3. Installment burden scatter + trend
            fig.add_trace(go.Scatter(x=df['installment_burden'], y=df['risk_binary'], mode='markers', name='Points'), row=1, col=3)
            z = np.polyfit(df['installment_burden'], df['risk_binary'], 1)
            p = np.poly1d(z)
            x_vals = np.sort(df['installment_burden'])
            fig.add_trace(go.Scatter(x=x_vals, y=p(x_vals), mode='lines', name='Trend'), row=1, col=3)

            # 4. Credit history trend
            hist = df.groupby('credit_history_score')['risk_binary'].mean().reset_index()
            fig.add_trace(go.Scatter(x=hist['credit_history_score'], y=hist['risk_binary']*100, mode='lines+markers'), row=2, col=1)

            # 5. Feature importance top 5
            if hasattr(self, "feature_importance_") and not self.feature_importance_.empty:
                top5 = self.feature_importance_.head(5)
                fig.add_trace(go.Bar(x=top5.values, y=[str(i) for i in top5.index], orientation='h', showlegend=False), row=2, col=2)

            # 6. Fraud pie
            fraud_counts = [int(len(df) - df['potential_fraud'].sum()), int(df['potential_fraud'].sum())]
            fig.add_trace(go.Pie(labels=['Normal', 'Potential Fraud'], values=fraud_counts, textinfo='percent+label'), row=2, col=3)

            fig.update_layout(height=900, width=1400, title_text="Credit Risk & Fraud Dashboard", showlegend=False, title_x=0.5)
            fig.write_html(html_path)
            logger.info("Interactive dashboard written to %s", html_path)

            # Try write image via kaleido if requested
            if try_kaleido:
                try:
                    fig.write_image(png_path, scale=2)
                    logger.info("PNG exported via plotly/kaleido to %s", png_path)
                    return {"html": html_path, "png": png_path}
                except Exception as e:
                    logger.warning("Plotly PNG export failed (kaleido missing?): %s", e)
        except Exception as exc:
            logger.exception("Plotly visualization failed: %s", exc)

        # Fallback: create static matplotlib-based cover PNG
        try:
            self._create_static_cover(png_path)
            logger.info("Static PNG cover created at %s", png_path)
            return {"html": html_path, "png": png_path}
        except Exception as e:
            logger.exception("Failed to create static cover PNG: %s", e)
            return {"html": html_path, "png": None}

    def _create_static_cover(self, path: str):
        """Create a single PNG that summarizes important metrics using matplotlib and PIL."""
        df = self.processed_data.copy()

        # Metrics
        total_apps = len(df)
        bad_rate = df['risk_binary'].mean() * 100
        fraud_rate = df['potential_fraud'].mean() * 100
        avg_loan = df['LoanAmount'].mean()

        # 1) header image (stats)
        fig, ax = plt.subplots(figsize=(12, 1.8))
        ax.axis("off")
        header_text = f"Total Applications: {total_apps:,}    Bad Rate: {bad_rate:.1f}%    Potential Fraud: {fraud_rate:.1f}%    Avg Loan: €{avg_loan:,.0f}"
        ax.text(0.01, 0.5, header_text, fontsize=14, va='center')
        header_png = path + "_hdr.png"
        fig.savefig(header_png, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # 2) loan amount histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['LoanAmount'], bins=30)
        ax.set_title("Loan Amount Distribution")
        ax.set_xlabel("Loan Amount")
        ax.set_ylabel("Count")
        hist_png = path + "_hist.png"
        fig.savefig(hist_png, bbox_inches="tight", dpi=120)
        plt.close(fig)

        # 3) top feature importance bar (matplotlib)
        fig, ax = plt.subplots(figsize=(6, 4))
        if hasattr(self, "feature_importance_") and not self.feature_importance_.empty:
            top5 = self.feature_importance_.head(5)[::-1]
            ax.barh(top5.index.astype(str), top5.values)
            ax.set_title("Top 5 Feature Importances")
        else:
            ax.text(0.1, 0.5, "No feature importance available", fontsize=12)
            ax.set_axis_off()
        feat_png = path + "_feat.png"
        fig.savefig(feat_png, bbox_inches="tight", dpi=120)
        plt.close(fig)

        # 4) Fraud pie
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = [int(len(df) - df['potential_fraud'].sum()), int(df['potential_fraud'].sum())]
        ax.pie(counts, labels=['Normal', 'Potential Fraud'], autopct="%1.1f%%", startangle=90)
        ax.set_title("Fraud Flags")
        pie_png = path + "_pie.png"
        fig.savefig(pie_png, bbox_inches="tight", dpi=120)
        plt.close(fig)

        # Compose final image with PIL
        img_hdr = Image.open(header_png).convert("RGB")
        img_hist = Image.open(hist_png).convert("RGB")
        img_feat = Image.open(feat_png).convert("RGB")
        img_pie = Image.open(pie_png).convert("RGB")

        # Resize to tidy layout
        width = max(img_hist.width + img_feat.width + 40, img_pie.width + 40, img_hdr.width)
        new_hdr = img_hdr.resize((width, int(img_hdr.height)))
        combined = Image.new("RGB", (width, new_hdr.height + max(img_hist.height, img_feat.height) + 40), (255, 255, 255))
        combined.paste(new_hdr, (0, 0))

        y = new_hdr.height + 20
        x = 20
        combined.paste(img_hist, (x, y))
        combined.paste(img_feat, (x + img_hist.width + 20, y))

        # paste pie to the right-bottom if room
        pie_x = width - img_pie.width - 20
        pie_y = y
        combined.paste(img_pie.resize((int(img_pie.width * 0.9), int(img_pie.height * 0.9))), (pie_x, pie_y))

        combined.save(path, dpi=(150, 150))
        # cleanup intermediate files
        for f in [header_png, hist_png, feat_png, pie_png]:
            try:
                os.remove(f)
            except Exception:
                pass
        return path

    # -------------------------
    # Reporting & utilities
    # -------------------------
    def generate_risk_report(self) -> str:
        if self.processed_data is None:
            return "No data available."

        df = self.processed_data
        total = len(df)
        bad_rate = df['risk_binary'].mean() * 100
        fraud_rate = df['potential_fraud'].mean() * 100
        avg_loan = df['LoanAmount'].mean()
        high_risk_threshold = df['risk_score'].quantile(0.8)
        high_risk = df[df['risk_score'] >= high_risk_threshold]

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
CREDIT RISK ASSESSMENT REPORT
Generated: {ts}

EXECUTIVE SUMMARY
 - Total applications: {total:,}
 - Overall bad credit rate: {bad_rate:.1f}%
 - Potential fraud cases: {fraud_rate:.1f}%
 - Average loan amount: €{avg_loan:,.0f}

KEY RISK FACTORS
 - Employment stability
 - Credit history quality
 - Installment burden
 - Savings level
 - Property ownership

HIGH RISK SEGMENT
 - High-risk threshold (80th percentile): {high_risk_threshold:.3f}
 - High-risk applications: {len(high_risk)} ({len(high_risk) / total * 100:.1f}%)
 - Bad credit rate in high-risk: {high_risk['risk_binary'].mean() * 100 if len(high_risk) else 0:.1f}%

FRAUD DETECTION
 - Detection method: IsolationForest
 - Cases flagged: {int(df['potential_fraud'].sum())}

RECOMMENDATIONS
 - Automated screening using financial stability score
 - Enhanced verification and manual review for flagged fraud cases
 - Periodic model retraining and drift monitoring
 - Audit logs for data transformations and model decisions
"""
        return report

    def save_model(self, path: str):
        if self.risk_model is None:
            logger.warning("No model to save.")
            return
        joblib.dump(self.risk_model, path)
        logger.info("Model saved to %s", path)


# -------------------------
# Command-line entrypoint
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Credit Risk & Fraud Detection Dashboard")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--quick", action="store_true", help="Quick train (no hyperparameter search)")
    parser.add_argument("--no-kaleido", action="store_true", help="Do not attempt plotly/kaleido PNG export")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save trained model (joblib)")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Starting analysis. Output directory: %s", args.output_dir)

    analyzer = CreditRiskAnalyzer()
    if args.data:
        ok = analyzer.load_data(args.data)
        if not ok:
            logger.error("Data load failed or data missing required columns. Exiting.")
            sys.exit(1)
    else:
        # generate sample data (same as previous)
        logger.info("No data provided. Generating sample dataset.")
        np.random.seed(42)
        sample_data = {
            'LoanAmount': np.random.normal(3000, 1500, 1000).clip(500, 20000).astype(int),
            'LoanDuration': np.random.choice([6, 12, 18, 24, 36, 48], 1000),
            'InstallmentPercent': np.random.choice([1, 2, 3, 4], 1000),
            'Age': np.random.normal(35, 12, 1000).clip(18, 80).astype(int),
            'EmploymentDuration': np.random.choice(
                ['unemployed', '<1 year', '1<=X<4 years', '4<=X<7 years', '>=7 years'], 1000),
            'CreditHistory': np.random.choice([
                'no credits taken/all credits paid back duly',
                'all credits at this bank paid back duly',
                'existing credits paid back duly till now',
                'delay in paying off in the past',
                'critical account/other credits existing'
            ], 1000),
            'ExistingSavings': np.random.choice(
                ['no known savings', '<100 DM', '100<=X<500 DM', '500<=X<1000 DM', '>=1000 DM'], 1000),
            'OwnsProperty': np.random.choice(
                ['no known property', 'car or other', 'building soc. savings agr./life insurance', 'real estate'],
                1000),
            'LoanPurpose': np.random.choice(
                ['education', 'new car', 'used car', 'furniture/equipment', 'business', 'repairs'], 1000),
            'CurrentResidenceDuration': np.random.choice([1, 2, 3, 4], 1000),
            'ExistingCreditsCount': np.random.choice([1, 2, 3, 4], 1000),
            'Risk': np.random.choice(['good', 'bad'], 1000, p=[0.7, 0.3])
        }
        analyzer.data = pd.DataFrame(sample_data)

    # pipeline
    analyzer.create_legitimate_features()
    analyzer.train_risk_model(quick=args.quick)
    analyzer.detect_anomalies()
    viz_paths = analyzer.create_dashboard_visualizations(output_dir=args.output_dir,
                                                        try_kaleido=not args.no_kaleido)
    report = analyzer.generate_risk_report()
    logger.info("\n%s", report)

    # save model if requested
    if args.save_model and analyzer.risk_model is not None:
        analyzer.save_model(args.save_model)

    logger.info("Finished. Visualizations: %s", viz_paths)
    print(report)


if __name__ == "__main__":
    main()
 # End
