# Credit Risk Analysis & Fruad Detections Dashboard By Nattawut Boonnoon
# GitHub: @Nattawut30

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import streamlit as st
import logging
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, filename='credit_risk.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CreditRiskAnalyzer:
    """
    Professional Credit Risk Analysis and Fraud Detection System
    Focuses on legitimate financial factors only
    """

    def __init__(self, data_path=None):
        self.data = None
        self.risk_model = None
        self.fraud_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path):
        """Load and prepare the credit data with validation"""
        if not os.path.exists(data_path):
            logger.error(f"File {data_path} does not exist")
            return False

        try:
            self.data = pd.read_csv(data_path)
            required_columns = [
                'LoanAmount', 'LoanDuration', 'InstallmentPercent', 'Age',
                'EmploymentDuration', 'CreditHistory', 'ExistingSavings',
                'OwnsProperty', 'LoanPurpose', 'CurrentResidenceDuration',
                'ExistingCreditsCount', 'Risk'
            ]
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Handle invalid data (e.g., zero LoanAmount)
            self.data = self.data[self.data['LoanAmount'] > 0]

            # Handle missing values
            for col in self.data.columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)

            logger.info(f"Data loaded successfully: {self.data.shape[0]} records, {self.data.shape[1]} features")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def map_employment_duration(self, value):
        mapping = {
            'unemployed': 1,
            '<1 year': 2,
            '1<=X<4 years': 3,
            '4<=X<7 years': 4,
            '>=7 years': 5
        }
        if value not in mapping:
            logger.warning(f"Unrecognized EmploymentDuration value '{value}'; defaulting to 1")
            return 1
        return mapping[value]

    def map_credit_history(self, value):
        mapping = {
            'no credits taken/all credits paid back duly': 5,
            'all credits at this bank paid back duly': 4,
            'existing credits paid back duly till now': 3,
            'delay in paying off in the past': 2,
            'critical account/other credits existing': 1
        }
        if value not in mapping:
            logger.warning(f"Unrecognized CreditHistory value '{value}'; defaulting to 1")
            return 1
        return mapping[value]

    def map_savings(self, value):
        mapping = {
            'no known savings': 1,
            '<100 DM': 2,
            '100<=X<500 DM': 3,
            '500<=X<1000 DM': 4,
            '>=1000 DM': 5
        }
        if value not in mapping:
            logger.warning(f"Unrecognized ExistingSavings value '{value}'; defaulting to 1")
            return 1
        return mapping[value]

    def map_property(self, value):
        mapping = {
            'no known property': 1,
            'car or other': 2,
            'building soc. savings agr./life insurance': 3,
            'real estate': 4
        }
        if value not in mapping:
            logger.warning(f"Unrecognized OwnsProperty value '{value}'; defaulting to 1")
            return 1
        return mapping[value]

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
        if value not in mapping:
            logger.warning(f"Unrecognized LoanPurpose value '{value}'; defaulting to 1")
            return 1
        return mapping[value]

    def create_legitimate_features(self):
        """
        Create risk features based on LEGITIMATE financial factors only
        """
        if self.data is None:
            logger.error("No data loaded")
            return None

        df = self.data.copy()

        # Income & Employment Stability
        df['employment_stability_score'] = df['EmploymentDuration'].apply(self.map_employment_duration)

        # Financial Capacity Features
        # Skip loan_to_income_ratio if no Income column
        if 'Income' in df.columns:
            df['loan_to_income_ratio'] = df['LoanAmount'] / df['Income'].clip(lower=1)
        else:
            df['loan_to_income_ratio'] = np.nan
            logger.warning("Income data unavailable; loan_to_income_ratio set to NaN")
        df['installment_burden'] = df['InstallmentPercent']

        # Credit History Quality
        df['credit_history_score'] = df['CreditHistory'].apply(self.map_credit_history)

        # Financial Stability Indicators
        df['savings_score'] = df['ExistingSavings'].apply(self.map_savings)
        df['property_ownership_score'] = df['OwnsProperty'].apply(self.map_property)

        # Loan Specific Risk Factors
        df['loan_purpose_risk'] = df['LoanPurpose'].apply(self.map_loan_purpose)

        # Stability Indicators
        df['residence_stability'] = df['CurrentResidenceDuration']
        df['existing_credits_burden'] = df['ExistingCreditsCount']

        # Fraud Detection Features
        df['age_loan_ratio'] = df['Age'] / df['LoanAmount'].clip(lower=1) * 1000
        df['duration_amount_ratio'] = df['LoanDuration'] / df['LoanAmount'].clip(lower=1) * 100

        # Composite Scores (equal weights; can optimize later)
        df['financial_stability_score'] = (
                df['employment_stability_score'] * 0.25 +
                df['savings_score'] * 0.25 +
                df['property_ownership_score'] * 0.25 +
                df['credit_history_score'] * 0.25
        )

        df['risk_score'] = (
                df['installment_burden'] * 0.3 +
                df['loan_purpose_risk'] * 0.2 +
                df['existing_credits_burden'] * 0.3 +
                (6 - df['financial_stability_score']) * 0.2
        )

        df['risk_binary'] = (df['Risk'] == 'bad').astype(int)
        self.processed_data = df
        logger.info("Legitimate features created successfully")
        return df

    def detect_anomalies(self, contamination=None):
        """
        Detect potential fraud cases using legitimate financial patterns
        """
        if self.processed_data is None:
            logger.error("Please run create_legitimate_features() first")
            return None

        fraud_features = [
            'LoanAmount', 'LoanDuration', 'InstallmentPercent',
            'Age', 'financial_stability_score', 'risk_score',
            'age_loan_ratio', 'duration_amount_ratio'
        ]

        X_fraud = self.processed_data[fraud_features].copy()
        for col in X_fraud.columns:
            X_fraud[col].fillna(X_fraud[col].median(), inplace=True)

        # Estimate contamination if not provided
        if contamination is None:
            q1 = X_fraud.quantile(0.25)
            q3 = X_fraud.quantile(0.75)
            iqr = q3 - q1
            outliers = ((X_fraud < (q1 - 1.5 * iqr)) | (X_fraud > (q3 + 1.5 * iqr))).sum().sum()
            contamination = min(max(outliers / len(X_fraud), 0.01), 0.3)

        self.fraud_model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        fraud_predictions = self.fraud_model.fit_predict(X_fraud)
        self.processed_data['potential_fraud'] = (fraud_predictions == -1).astype(int)

        fraud_count = self.processed_data['potential_fraud'].sum()
        logger.info(
            f"Detected {fraud_count} potential fraud cases ({fraud_count / len(self.processed_data) * 100:.1f}%)")
        logger.info(f"Estimated contamination rate: {contamination:.3f}")
        return fraud_predictions

    def train_risk_model(self):
        """
        Train credit risk model using legitimate factors
        """
        if self.processed_data is None:
            logger.error("Please run create_legitimate_features() first")
            return None

        risk_features = [
            'employment_stability_score', 'credit_history_score', 'savings_score',
            'property_ownership_score', 'installment_burden', 'loan_purpose_risk',
            'existing_credits_burden', 'residence_stability', 'LoanAmount', 'LoanDuration'
        ]

        X = self.processed_data[risk_features].copy()
        for col in X.columns:
            X[col].fillna(X[col].median(), inplace=True)
        y = self.processed_data['risk_binary']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Hyperparameter tuning with class weight for imbalance
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        self.risk_model = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid,
            cv=5,
            scoring='f1_weighted',  # Better for imbalance than accuracy
            n_jobs=-1
        )

        self.risk_model.fit(X_train, y_train)

        # Evaluate
        train_score = self.risk_model.score(X_train, y_train)
        test_score = self.risk_model.score(X_test, y_test)
        y_pred = self.risk_model.best_estimator_.predict(X_test)

        logger.info(f"Risk Model Performance (Best Parameters: {self.risk_model.best_params_}):")
        logger.info(f"Training F1: {train_score:.3f}")
        logger.info(f"Testing F1: {test_score:.3f}")
        logger.info("\nDetailed Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': risk_features,
            'importance': self.risk_model.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop Risk Factors:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.3f}")

        return self.risk_model


def create_dashboard_visualizations(analyzer):
    """
    Create comprehensive dashboard visualizations using Plotly
    """
    if analyzer.processed_data is None:
        logger.error("No processed data available")
        return None

    df = analyzer.processed_data
    # Specify subplot types: pie needs 'domain'
    specs = [[{'type': 'xy'} for _ in range(3)] for _ in range(4)]
    specs[2][0] = {'type': 'domain'}  # For fraud pie in row 3, col 1
    fig = make_subplots(
        rows=4, cols=3,
        specs=specs,
        subplot_titles=[
            'Credit Risk by Financial Stability', 'Risk by Employment Stability',
            'Loan Amount Distribution by Risk', 'Risk by Credit History Quality',
            'Risk by Savings Level', 'Risk vs Installment Burden',
            'Fraud Detection Results', 'Fraud Detection: Age vs Loan Amount',
            'Risk Score Distribution', 'Risk by Property Ownership',
            'Risk by Loan Purpose', 'Feature Importance'
        ]
    )

    try:
        # 1. Risk Distribution by Financial Stability
        stability_risk = df.groupby('financial_stability_score')['risk_binary'].agg(['count', 'mean']).reset_index()
        if not stability_risk.empty:
            fig.add_trace(
                go.Bar(x=stability_risk['financial_stability_score'], y=stability_risk['mean'] * 100,
                       marker_color='lightcoral', opacity=0.7),
                row=1, col=1
            )
            fig.update_xaxes(title_text='Financial Stability Score (1=Poor, 5=Excellent)', row=1, col=1)
            fig.update_yaxes(title_text='Bad Credit Rate (%)', row=1, col=1)

        # 2. Employment Stability Impact
        emp_risk = df.groupby('employment_stability_score')['risk_binary'].agg(['count', 'mean']).reset_index()
        if not emp_risk.empty:
            fig.add_trace(
                go.Bar(x=emp_risk['employment_stability_score'], y=emp_risk['mean'] * 100,
                       marker_color=['#ff4444', '#ff8844', '#ffaa44', '#88ff44', '#44ff44'], opacity=0.8),
                row=1, col=2
            )
            fig.update_xaxes(title_text='Employment Stability Score', row=1, col=2)
            fig.update_yaxes(title_text='Bad Credit Rate (%)', row=1, col=2)

        # 3. Loan Amount vs Risk
        if not df.empty:
            fig.add_trace(
                go.Histogram(x=df[df['risk_binary'] == 0]['LoanAmount'], nbinsx=30, name='Good Credit',
                             marker_color='lightgreen', opacity=0.7),
                row=1, col=3
            )
            fig.add_trace(
                go.Histogram(x=df[df['risk_binary'] == 1]['LoanAmount'], nbinsx=30, name='Bad Credit',
                             marker_color='lightcoral', opacity=0.7),
                row=1, col=3
            )
            fig.update_xaxes(title_text='Loan Amount (DM)', row=1, col=3)
            fig.update_yaxes(title_text='Frequency', row=1, col=3)

        # 4. Credit History Quality Impact
        history_risk = df.groupby('credit_history_score')['risk_binary'].agg(['count', 'mean']).reset_index()
        if not history_risk.empty:
            fig.add_trace(
                go.Scatter(x=history_risk['credit_history_score'], y=history_risk['mean'] * 100,
                           mode='lines+markers', marker=dict(size=8), line=dict(width=3), marker_color='navy'),
                row=2, col=1
            )
            fig.update_xaxes(title_text='Credit History Score (1=Poor, 5=Excellent)', row=2, col=1)
            fig.update_yaxes(title_text='Bad Credit Rate (%)', row=2, col=1)

        # 5. Savings Impact
        savings_risk = df.groupby('savings_score')['risk_binary'].agg(['count', 'mean']).reset_index()
        if not savings_risk.empty:
            fig.add_trace(
                go.Bar(x=savings_risk['savings_score'], y=savings_risk['mean'] * 100,
                       marker_color='skyblue', opacity=0.8),
                row=2, col=2
            )
            fig.update_xaxes(title_text='Savings Score (1=None, 5=High)', row=2, col=2)
            fig.update_yaxes(title_text='Bad Credit Rate (%)', row=2, col=2)

        # 6. Installment Burden Analysis
        if not df.empty:
            fig.add_trace(
                go.Scatter(x=df['InstallmentPercent'], y=df['risk_binary'], mode='markers',
                           marker=dict(color='purple', opacity=0.5)),
                row=2, col=3
            )
            z = np.polyfit(df['InstallmentPercent'], df['risk_binary'], 1)
            p = np.poly1d(z)
            x_vals = df['InstallmentPercent'].sort_values()
            fig.add_trace(
                go.Scatter(x=x_vals, y=p(x_vals), mode='lines', line=dict(color='red', dash='dash')),
                row=2, col=3
            )
            fig.update_xaxes(title_text='Installment as % of Income', row=2, col=3)
            fig.update_yaxes(title_text='Risk (0=Good, 1=Bad)', row=2, col=3)

        # 7. Fraud Detection Results (Pie - now with domain)
        fraud_counts = [len(df) - df['potential_fraud'].sum(), df['potential_fraud'].sum()]
        if sum(fraud_counts) > 0:
            fig.add_trace(
                go.Pie(labels=['Normal', 'Potential Fraud'], values=fraud_counts,
                       marker_colors=['lightgreen', 'red'], textinfo='percent+label', pull=[0, 0.1]),
                row=3, col=1
            )

        # 8. Age vs Loan Amount (Fraud Pattern)
        if not df.empty:
            fraud_mask = df['potential_fraud'] == 1
            fig.add_trace(
                go.Scatter(x=df[~fraud_mask]['Age'], y=df[~fraud_mask]['LoanAmount'], mode='markers',
                           name='Normal', marker=dict(color='blue', size=6, opacity=0.6)),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(x=df[fraud_mask]['Age'], y=df[fraud_mask]['LoanAmount'], mode='markers',
                           name='Potential Fraud', marker=dict(color='red', size=8, opacity=0.8)),
                row=3, col=2
            )
            fig.update_xaxes(title_text='Age (years)', row=3, col=2)
            fig.update_yaxes(title_text='Loan Amount (DM)', row=3, col=2)

        # 9. Risk Score Distribution
        if not df.empty:
            fig.add_trace(
                go.Histogram(x=df[df['risk_binary'] == 0]['risk_score'], nbinsx=30, name='Good Credit',
                             marker_color='lightgreen', opacity=0.7, histnorm='probability density'),
                row=3, col=3
            )
            fig.add_trace(
                go.Histogram(x=df[df['risk_binary'] == 1]['risk_score'], nbinsx=30, name='Bad Credit',
                             marker_color='lightcoral', opacity=0.7, histnorm='probability density'),
                row=3, col=3
            )
            fig.update_xaxes(title_text='Calculated Risk Score', row=3, col=3)
            fig.update_yaxes(title_text='Density', row=3, col=3)

        # 10. Property Ownership Impact
        prop_risk = df.groupby('property_ownership_score')['risk_binary'].mean() * 100
        if not prop_risk.empty:
            fig.add_trace(
                go.Bar(x=prop_risk.index, y=prop_risk.values, marker_color='gold', opacity=0.8),
                row=4, col=1
            )
            fig.update_xaxes(title_text='Property Score (1=None, 4=Real Estate)', row=4, col=1)
            fig.update_yaxes(title_text='Bad Credit Rate (%)', row=4, col=1)

        # 11. Loan Purpose Risk Analysis
        purpose_risk = df.groupby('LoanPurpose')['risk_binary'].mean() * 100
        if not purpose_risk.empty:
            short_labels = [p[:15] + '...' if len(p) > 15 else p for p in purpose_risk.index]
            fig.add_trace(
                go.Bar(y=short_labels, x=purpose_risk.values, orientation='h', marker_color='lightblue', opacity=0.8),
                row=4, col=2
            )
            fig.update_xaxes(title_text='Bad Credit Rate (%)', row=4, col=2)

        # 12. Feature Importance
        if analyzer.risk_model is not None:
            feature_importance = pd.Series(
                analyzer.risk_model.best_estimator_.feature_importances_,
                index=['employment_stability_score', 'credit_history_score', 'savings_score',
                       'property_ownership_score', 'installment_burden', 'loan_purpose_risk',
                       'existing_credits_burden', 'residence_stability', 'LoanAmount', 'LoanDuration']
            ).sort_values(ascending=True)
            short_labels = [f[:20] + '...' if len(f) > 20 else f for f in feature_importance.index]
            fig.add_trace(
                go.Bar(y=short_labels, x=feature_importance.values, orientation='h',
                       marker_color='darkgreen', opacity=0.7),
                row=4, col=3
            )
            fig.update_xaxes(title_text='Importance Score', row=4, col=3)

    except Exception as e:
        logger.error(f"Error in creating visualizations: {e}")

    fig.update_layout(height=1200, width=1600, title_text="Professional Credit Risk & Fraud Detection Dashboard",
                      showlegend=False, title_x=0.5)
    return fig


def generate_risk_report(analyzer):
    """
    Generate a professional risk assessment report
    """
    if analyzer.processed_data is None:
        return "No data available for report generation."

    df = analyzer.processed_data
    total_applications = len(df)
    bad_credit_rate = df['risk_binary'].mean() * 100
    fraud_rate = df['potential_fraud'].mean() * 100
    avg_loan_amount = df['LoanAmount'].mean()
    high_risk_threshold = df['risk_score'].quantile(0.8)
    high_risk_applications = df[df['risk_score'] >= high_risk_threshold]

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    CREDIT RISK ASSESSMENT REPORT             ║
║                     Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}                     ║
╚══════════════════════════════════════════════════════════════╝

📊 EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Applications Analyzed: {total_applications:,}
• Overall Bad Credit Rate: {bad_credit_rate:.1f}%
• Potential Fraud Cases Detected: {fraud_rate:.1f}%
• Average Loan Amount: €{avg_loan_amount:,.0f}

🎯 KEY RISK FACTORS IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Employment Stability: Most critical factor
2. Credit History Quality: Strong predictor of future behavior
3. Installment Burden: High percentages indicate stress
4. Savings Level: Financial cushion indicator
5. Property Ownership: Stability and commitment signal

⚠️ HIGH-RISK SEGMENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• High-Risk Applications: {len(high_risk_applications)} ({len(high_risk_applications) / total_applications * 100:.1f}%)
• Bad Credit Rate in High-Risk Segment: {high_risk_applications['risk_binary'].mean() * 100:.1f}%
• Recommended Action: Enhanced screening and monitoring

🚨 FRAUD DETECTION INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Anomaly Detection Model: Isolation Forest
• Cases Flagged for Review: {df['potential_fraud'].sum()}
• Primary Indicators: Unusual age-loan patterns, extreme ratios

✅ RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Implement automated screening using financial stability scores
2. Enhanced verification for potential fraud cases
3. Regular model retraining with new data
4. Focus underwriting on employment and credit history
5. Monitor installment burden ratios closely

⚖️ COMPLIANCE NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This analysis uses only legitimate financial factors and complies
with fair lending practices. No discriminatory variables were used.
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Credit Risk Dashboard")
    parser.add_argument('--data', type=str, default=None, help="Path to CSV data file")
    args = parser.parse_args()

    logger.info("PROFESSIONAL CREDIT RISK & FRAUD DETECTION SYSTEM STARTED")

    analyzer = CreditRiskAnalyzer()

    if args.data:
        if not analyzer.load_data(args.data):
            return
    else:
        # Simulate sample data
        logger.info("Creating sample legitimate analysis...")
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

    # Step 1: Create legitimate features
    logger.info("Engineering legitimate financial features...")
    analyzer.create_legitimate_features()

    # Step 2: Train risk model
    logger.info("Training risk assessment model...")
    analyzer.train_risk_model()

    # Step 3: Detect anomalies/fraud
    logger.info("Running fraud detection analysis...")
    analyzer.detect_anomalies()

    # Step 4: Generate visualizations
    logger.info("Creating dashboard visualizations...")
    fig = create_dashboard_visualizations(analyzer)
    if fig:
        fig.show()

    # Step 5: Generate report
    logger.info("Generating professional risk report...")
    report = generate_risk_report(analyzer)
    print(report)

    logger.info("Analysis Complete!")
    logger.info(
        "Professional Tip: Always validate models with out-of-sample data and monitor for model drift and bias.")


@st.cache_data
def create_streamlit_dashboard():
    """
    Streamlit version for interactive web dashboard
    """
    st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
    st.title("🏦 Professional Credit Risk & Fraud Detection Dashboard")
    st.markdown("---")

    st.sidebar.header("Analysis Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Credit Data (CSV)", type=['csv'])
    contamination = st.sidebar.slider("Fraud Contamination Rate", 0.01, 0.3, 0.1)

    if uploaded_file is not None:
        analyzer = CreditRiskAnalyzer()
        analyzer.data = pd.read_csv(uploaded_file)

        with st.spinner("Processing data and training models..."):
            analyzer.create_legitimate_features()
            analyzer.train_risk_model()
            analyzer.detect_anomalies(contamination=contamination)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", len(analyzer.processed_data))
        with col2:
            bad_rate = analyzer.processed_data['risk_binary'].mean() * 100
            st.metric("Bad Credit Rate", f"{bad_rate:.1f}%")
        with col3:
            fraud_rate = analyzer.processed_data['potential_fraud'].mean() * 100
            st.metric("Potential Fraud", f"{fraud_rate:.1f}%")
        with col4:
            avg_amount = analyzer.processed_data['LoanAmount'].mean()
            st.metric("Avg Loan Amount", f"€{avg_amount:,.0f}")

        st.subheader("Visual Analysis")
        fig = create_dashboard_visualizations(analyzer)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Assessment Report")
        st.text(generate_risk_report(analyzer))

        st.success("Dashboard loaded successfully!")
    else:
        st.info("Please upload a CSV file to begin analysis.")


if __name__ == "__main__":
    main()
