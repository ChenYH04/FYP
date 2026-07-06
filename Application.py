import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

# ----------------------------------------
# 1. Page Setup
# ----------------------------------------
st.set_page_config(page_title="Telecom Churn Prediction System", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.jpeg", width=600)

st.title("Telecom Churn Prediction System")
st.markdown("Upload customer data to view churn predictions, financial impact, and retention strategies.")

# ----------------------------------------
# Sidebar Controls (NEW)
# ----------------------------------------
st.sidebar.title("⚙️ Control Panel")

# ----------------- Model Selection -------------------
st.sidebar.markdown("### 🤖 Model Selection")

model_choice = st.sidebar.radio(
    "Choose Prediction Model",
    ["Random Forest", "Logistic Regression"],
)

# ---------------- Risk Thresholds (NEW FLEX CONTROL) ----------------
st.sidebar.markdown("### 🎯 Risk Thresholds")

high_risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.5,
    max_value=0.9,
    value=0.7,
    step=0.05
)

medium_risk_threshold = st.sidebar.slider(
    "Medium Risk Threshold",
    min_value=0.2,
    max_value=0.6,
    value=0.4,
    step=0.05
)

# ---------------- Financial Controls ----------------
st.sidebar.markdown("### 💰 Financial Settings")

budget = st.sidebar.slider(
    "Retention Budget ($)",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=500
)

# ---------------- UI Toggles ----------------
st.sidebar.markdown("### 📊 Display Options")

show_high_risk_table = st.sidebar.checkbox("Show High-Risk Table", value=True)
show_financial_dashboard = st.sidebar.checkbox("Show Financial Dashboard", value=True)
show_shap_explanations = st.sidebar.checkbox("Show SHAP Explanations", value=True)


st.sidebar.markdown("---")
st.sidebar.caption("Telecom Churn System v2.0")

# ----------------------------------------
# 2. Data Upload Panel
# ----------------------------------------
st.header("Upload Customer Data")
uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])

required_columns = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # ── PREPROCESSING 1: Remove duplicate customerIDs ────────────────────────
    data = data.drop_duplicates(subset=["customerID"], keep="first")

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success("Data validated successfully!")
    
        # ----------------------------------------
        # Exploratory Data Analysis (EDA)
        # ----------------------------------------
        with st.expander("📊 Exploratory Data Analysis", expanded=False):
            st.subheader("Missing Value Summary")

            missing_summary = data.isnull().sum()
            missing_pct     = (missing_summary / len(data) * 100).round(2)
            missing_df = pd.DataFrame({
                "Column":            missing_summary.index,
                "Missing Count":     missing_summary.values,
                "Missing %":         missing_pct.values,
            }).sort_values("Missing Count", ascending=False)

            cols_with_missing = missing_df[missing_df["Missing Count"] > 0]
            if len(cols_with_missing) > 0:
                st.write(f"**{len(cols_with_missing)} column(s) contain missing values:**")
                st.dataframe(cols_with_missing, width="stretch", hide_index=True)
            else:
                st.success("No missing values detected in the raw uploaded dataset.")

            st.subheader("Descriptive Statistics (Numeric Columns)")
            st.dataframe(data.describe(), width="stretch")

            st.subheader("Target Variable Distribution (Raw)")
            raw_churn_counts = data["Churn"].astype(str).str.strip().str.lower().value_counts()
            fig_eda_churn = px.bar(
                x=raw_churn_counts.index,
                y=raw_churn_counts.values,
                labels={"x": "Churn (raw label)", "y": "Count"},
                title="Raw Churn Label Distribution Before Cleaning",
                color=raw_churn_counts.values,
                color_continuous_scale="Reds",
            )
            fig_eda_churn.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_eda_churn, width="stretch")

            imbalance_ratio = raw_churn_counts.max() / raw_churn_counts.min() if len(raw_churn_counts) > 1 else None
            if imbalance_ratio is not None:
                st.caption(
                    f"⚖️ Class imbalance ratio (majority:minority) ≈ {imbalance_ratio:.2f}:1 — "
                    "this informs the use of SMOTE during model training below."
                )

# -------------------------------------------------------------
# 3. Churn Prediction with Random Forest or Logistic Regression
# -------------------------------------------------------------
        st.header("Churn Prediction Results")

        # ── Clean core columns: required for CLV/RCC financial calculations ──
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data = data.dropna(subset=["TotalCharges", "tenure", "MonthlyCharges"])

        # ── Feature Engineering ───────────────────────────────────────────────
        data["AvgMonthlyCharge"]    = data["TotalCharges"] / data["tenure"].replace(0, np.nan)
        data["ChargeIncreaseRatio"] = data["MonthlyCharges"] / data["AvgMonthlyCharge"].replace(0, np.nan)
        data["ValueDensity"]        = data["TotalCharges"] / (data["tenure"] + 1)

        data["TenureBucket"] = pd.cut(
            data["tenure"],
            bins=[0, 6, 12, 24, 48, np.inf],
            labels=["0-6m", "6-12m", "1-2yr", "2-4yr", "4yr+"]
        )
        data["SpendTier"] = pd.qcut(
            data["MonthlyCharges"],
            q=4,
            labels=["Low", "Mid", "High", "Premium"]
        )

        # ── PREPROCESSING 2: IQR outlier capping ─────────────────────────────
        for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
            Q1  = data[col].quantile(0.25)
            Q3  = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data[col] = data[col].clip(
                lower=Q1 - 1.5 * IQR,
                upper=Q3 + 1.5 * IQR
            )

        # ── Normalize Churn label ────────────────────────────────────────────
        churn_yes_values = {"yes", "churn", "churned", "1", "true", "y"}
        data["Churn"] = data["Churn"].astype(str).str.strip().str.lower().apply(
            lambda v: "Yes" if v in churn_yes_values else "No"
        )
        data["ChurnEncoded"] = (data["Churn"] == "Yes").astype(int)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Dataset Summary")
        
        if uploaded_file and "ChurnEncoded" in data.columns:
            churn_count = data["ChurnEncoded"].sum()
            total_count = len(data)
            churn_rate  = churn_count / total_count * 100

            st.sidebar.metric(
                label="Churn Rate in Dataset",
                value=f"{churn_rate:.1f}%",
                delta=f"{churn_count:,} of {total_count:,} customers",
                delta_color="off",
            )
            st.sidebar.progress(int(churn_rate))
        else:
            st.sidebar.info("Upload a CSV to see churn rate.")

        # ── Use ALL available numeric/categorical features ───────────────────
        # Identify categorical columns (excluding identifiers and target)
        exclude_cols = {"customerID", "Churn", "ChurnEncoded"}
        cat_cols = [
            c for c in data.select_dtypes(include=["object", "category"]).columns
            if c not in exclude_cols
        ]
        num_cols = [
            c for c in data.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols and c != "ChurnEncoded"
        ]

        # ── PREPROCESSING 3: Rare category grouping ──────────────────────────
        for col in cat_cols:
            freq = data[col].value_counts(normalize=True)
            rare = freq[freq < 0.01].index
            data[col] = data[col].apply(lambda x: "__rare__" if x in rare else x)

        # One-hot encode categoricals
        encoded_data = data.copy()
        ohe_df = pd.get_dummies(data[cat_cols].astype(str), drop_first=True)

        # Reset data index first so data, X, and y all share the same 0-based index
        data = data.reset_index(drop=True)

        feature_cols = num_cols + list(ohe_df.columns)
        
        X = pd.concat(
            [data[num_cols].fillna(0).reset_index(drop=True),
             ohe_df.reset_index(drop=True)],
            axis=1
        )
        
        y = data["ChurnEncoded"]

        # ── Class imbalance: stratified split + balanced weights ─────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42, k_neighbors=5,)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=12,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )
        
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.1,
                max_iter=1000,
                class_weight="balanced",
            )

            # ── Targeted interaction terms for Logistic Regression ───────────────
            interaction_pairs = [
                ("tenure",          "MonthlyCharges"),
                ("tenure",          "TotalCharges"),
                ("MonthlyCharges",  "ValueDensity"),
            ]

            new_interaction_cols = []
            for col_a, col_b in interaction_pairs:
                if col_a in X_train.columns and col_b in X_train.columns:
                    col_name = f"{col_a}_x_{col_b}"
                    X_train[col_name] = X_train[col_a] * X_train[col_b]
                    X_test[col_name]  = X_test[col_a]  * X_test[col_b]
                    X[col_name]       = X[col_a]        * X[col_b]
                    new_interaction_cols.append(col_name)

            feature_cols = feature_cols + new_interaction_cols

            # ── Feature selection for Logistic Regression ────────────────────────
            selector = VarianceThreshold(threshold=0.01) 
            X_train_arr = selector.fit_transform(X_train)
            X_test_arr  = selector.transform(X_test)
            X_arr       = selector.transform(X)
            
            selected_cols = [feature_cols[i] for i in selector.get_support(indices=True)]
            X_train       = pd.DataFrame(X_train_arr, columns=selected_cols)
            X_test        = pd.DataFrame(X_test_arr,  columns=selected_cols)
            X             = pd.DataFrame(X_arr,       columns=selected_cols)
            feature_cols  = selected_cols
                    
            # ── PREPROCESSING 5: StandardScaler for Logistic Regression only ─
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feature_cols)
            X       = pd.DataFrame(scaler.transform(X),       columns=feature_cols)
        

        # Train model
        model.fit(X_train, y_train)
        
        # Predict probabilities
        data["ChurnProbability"] = model.predict_proba(X)[:, 1]

        # ── Risk segmentation ───────────────────────────────────────────────
        def risk_label(p):
            if p >= high_risk_threshold:
                return "🔴 High Risk"
            elif p >= medium_risk_threshold:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"

        data["RiskSegment"] = data["ChurnProbability"].apply(risk_label)

        # ── Summary KPIs ────────────────────────────────────────────────────
        total_customers = len(data)
        high_risk_count = (data["ChurnProbability"] >= high_risk_threshold).sum()
        med_risk_count  = ((data["ChurnProbability"] >= medium_risk_threshold) & (data["ChurnProbability"] < high_risk_threshold)).sum()
        low_risk_count  = (data["ChurnProbability"] < medium_risk_threshold).sum()
        pct_high = high_risk_count / total_customers * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers",   f"{total_customers:,}")
        k2.metric("🔴 High Risk",       f"{high_risk_count:,}",  f"{pct_high:.1f}% of base")
        k3.metric("🟡 Medium Risk",     f"{med_risk_count:,}")
        k4.metric("🟢 Low Risk",        f"{low_risk_count:,}")

        # ── Churn probability distribution ──────────────────────────────────
        bins = np.linspace(0, 1, 21)
        bin_labels = [f"{round(bins[i], 2)}–{round(bins[i+1], 2)}" for i in range(len(bins)-1)]
        data["ChurnBin"] = pd.cut(data["ChurnProbability"], bins=bins, labels=bin_labels, include_lowest=True)
        bin_counts = data["ChurnBin"].value_counts().sort_index()
        st.write("**Churn Probability Distribution (Binned):**")
        st.bar_chart(bin_counts)

        # ── High-risk table ──────────────────────────────────────────────────
        if show_high_risk_table:
            high_risk = data[data["ChurnProbability"] >= high_risk_threshold].sort_values(
                "ChurnProbability", ascending=False
            )

            st.write(
                f"**High-Risk Customers (≥ {high_risk_threshold:.2f}) — {len(high_risk)} customers:**"
            )

            st.dataframe(
                high_risk[
                    ["customerID", "ChurnProbability", "RiskSegment",
                     "MonthlyCharges", "TotalCharges", "Churn"]
                ],
                width="stretch",
                hide_index=True,
            )

        # ── CSV export ───────────────────────────────────────────────────────
        export_df = data[["customerID", "Churn", "ChurnProbability", "RiskSegment",
                           "tenure", "MonthlyCharges", "TotalCharges"]].copy()
        export_df["ChurnProbability"] = export_df["ChurnProbability"].round(4)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Predictions CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

# ----------------------------------------
# 4. Financial Impact Dashboard
# ----------------------------------------
        data["CLV_Calc"]    = data["MonthlyCharges"] * data["tenure"]
        data["tenure_safe"] = data["tenure"].replace(0, np.nan)
        data["RCC"]         = data["TotalCharges"] / data["tenure_safe"]
        
        if show_financial_dashboard:
            st.header("Financial Impact")
            
            clv_at_risk = data.loc[
                data["ChurnProbability"] >= high_risk_threshold,
                "CLV_Calc"
            ].sum()

            clv_total = data["CLV_Calc"].sum()

            avg_clv_churner = data.loc[data["Churn"] == "Yes", "CLV_Calc"].mean()

            f1, f2, f3 = st.columns(3)
            f1.metric("Total CLV", f"${clv_total:,.0f}")
            f2.metric(
                "CLV at Risk",
                f"${clv_at_risk:,.0f}",
                f"{clv_at_risk/clv_total*100:.1f}%"
            )
            f3.metric(
                "Avg CLV of Churners",
                f"${avg_clv_churner:,.0f}" if not np.isnan(avg_clv_churner) else "N/A"
            )

# ----------------------------------------
# 5. Retention Strategy Panel
# ----------------------------------------
        # ── Compute Value at Risk per customer ───────────────────────────────
        # Value at Risk = how much CLV we expect to lose from this customer
        # = Churn Probability × CLV (expected loss, not worst-case)
        data["ValueAtRisk"] = data["ChurnProbability"] * data["CLV_Calc"]

        # ── Classify customers into retention tiers ──────────────────────────
        # Priority:  High churn risk (≥60%) AND high CLV (above median) → best ROI
        # Standard:  High churn risk OR high CLV, but not both
        # Passive:   Low churn risk AND low CLV → minimal spend warranted
        clv_median = data["CLV_Calc"].median()

        def classify_tier(row):
            high_risk = row["ChurnProbability"] >= high_risk_threshold
            high_clv  = row["CLV_Calc"] >= clv_median
            if high_risk and high_clv:
                return "🔴 Priority"
            elif high_risk or high_clv:
                return "🟡 Standard"
            else:
                return "🟢 Passive"

        data["RetentionTier"] = data.apply(classify_tier, axis=1)

# ----------------------------------------
# 🔍 SHAP Explanation
# ----------------------------------------
        if show_shap_explanations:
            st.header("Churn Drivers (Dataset-Level Insights)")
        
            st.caption("This analysis explains the **main contributors to customers churn across the entire dataset**, not just one individual.")
        
            # ---------------- SHAP Calculation (HIGH-RISK FOCUSED) ----------------
            # Focus SHAP on high-risk customers
            high_risk_data = data[data["ChurnProbability"] >= high_risk_threshold]

            # fallback if too small
            if len(high_risk_data) > 50:
                sample_df = high_risk_data.sample(min(200, len(high_risk_data)), random_state=42)
            else:
                sample_df = data.sample(min(200, len(data)), random_state=42)

            # Align with model features
            X_sample = X.iloc[data.index.get_indexer(sample_df.index)]

            if model_choice in ["Random Forest"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample, approximate=True)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]

            else:
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_sample)

            # Final safety check
            if shap_values.shape[1] != len(feature_cols):
                raise ValueError(
                    f"SHAP shape mismatch: {shap_values.shape} vs {len(feature_cols)} features"
                )

            # Convert to DataFrame
            shap_df = pd.DataFrame(shap_values, columns=feature_cols)
        
            # ---------------- Chart ----------------
            # Calculate mean absolute SHAP importance
            top_features = pd.Series(
                np.abs(shap_df).mean(axis=0),
                index=feature_cols
            ).sort_values(ascending=False).head(10)

            fig_global = px.bar(
                x=top_features.values,
                y=top_features.index,
                orientation="h",
                title="Top Factors Driving Churn (Overall Dataset)",
                labels={"x": "Average Impact on Churn", "y": "Feature"},
                color=top_features.values,
                color_continuous_scale="Reds"
            )
        
            fig_global.update_layout(height=450)
            st.plotly_chart(fig_global, width="stretch")
        
            # ---------------- Churn Drivers (COMBINED & FOCUSED) ----------------
            st.subheader("🔥 Key Drivers of Customer Churn (High-Risk Focus)")
            st.caption("Only factors that actively push customers toward churn are shown below.")

            driver_rows = []

            for feature in top_features.index:
                shap_mean = shap_df[feature].mean()
                shap_strength = np.abs(shap_df[feature]).mean()

                # ✅ Only include churn-driving features
                if shap_mean <= 0:
                    continue
                
                clean_feature = feature.replace("_enc", "")

                # Data-driven explanation derived from actual feature distributions
                if feature in data.columns:
                    feature_mean   = data[feature].mean()
                    high_risk_mean = data.loc[
                        data["ChurnProbability"] >= high_risk_threshold, feature
                    ].mean()

                    if pd.isna(high_risk_mean) or pd.isna(feature_mean):
                        explanation = f"{clean_feature} is a contributing factor to churn risk in this dataset."
                    elif high_risk_mean < feature_mean:
                        explanation = (
                            f"High-risk customers have a lower average {clean_feature} "
                            f"({high_risk_mean:.2f}) compared to the overall average "
                            f"({feature_mean:.2f}), suggesting lower values of this "
                            f"feature are associated with increased churn risk."
                        )
                    else:
                        explanation = (
                            f"High-risk customers have a higher average {clean_feature} "
                            f"({high_risk_mean:.2f}) compared to the overall average "
                            f"({feature_mean:.2f}), suggesting higher values of this "
                            f"feature are associated with increased churn risk."
                        )
                else:
                    # Fallback for OHE features not directly in data.columns
                    explanation = (
                        f"{clean_feature} is a contributing factor to churn risk "
                        f"based on SHAP analysis of high-risk customers."
                    )
                    
                driver_rows.append({
                    "Driver": clean_feature,
                    "Impact Strength": round(shap_strength, 4),
                    "Avg SHAP Impact": round(shap_mean, 4),
                    "Business Insight": explanation
                })

            driver_df = pd.DataFrame(driver_rows).sort_values("Impact Strength", ascending=False)

            st.dataframe(driver_df, width="stretch", hide_index=True)

            st.caption("Avg SHAP Impact > 0 indicates the feature is actively pushing customers toward churn.")

            # ---------------- Executive Summary (FOCUSED) ----------------
            st.subheader("📢 Executive Summary")

            top_churn_drivers = [
                f.replace("_enc", "")
                for f in top_features.index
                if shap_df[f].mean() > 0
            ][:3]

            if len(top_churn_drivers) == 0:
                st.info("No strong churn-driving factors identified.")
            else:
                driver_text = ", ".join([f"**{f}**" for f in top_churn_drivers])

                summary_text = f"""
                • The primary factors driving customer churn are {driver_text}.

                • These factors are actively increasing churn probability among high-risk customers.

                • Business implication:
                  - These drivers should be the **top priority for intervention**
                  - Addressing them will deliver the **highest impact on churn reduction**

                • Recommended focus:
                  - Target customers exposed to these risk factors
                  - Design retention actions specifically to counter these drivers
                """

                st.info(summary_text)

        # ── Scenario definitions ─────────────────────────────────────────────
        st.header("Retention Strategy Recommendations")

        scenario_budget_ranges = {
            "🟡  Balanced Approach":          (3000,  7000, 5000),
            "🟢  Cost-Efficient":             (1000,  4000, 2000),
            "🔵  High-Value Customer Focus":  (5000, 10000, 7000),
            "🤖  Automation-First":           (1000, 10000, 5000),
        }

        option_a_scenarios = {"⚡  Win-Back Campaign", "🤖  Automation-First", "🤝  Loyalty Builder", "🌐  Broad Engagement"}

        # How each scenario splits total budget across the three customer tiers.
        # These weights drive WHERE the money goes before strategy tactics are applied.
        scenario_tier_weights = {
            "🟡  Balanced Approach":          {"🔴 Priority": 0.55, "🟡 Standard": 0.35, "🟢 Passive": 0.10},
            "🟢  Cost-Efficient":             {"🔴 Priority": 0.40, "🟡 Standard": 0.40, "🟢 Passive": 0.20},
            "🔵  High-Value Customer Focus":  {"🔴 Priority": 0.85, "🟡 Standard": 0.15, "🟢 Passive": 0.00},
            "🤖  Automation-First":           {"🔴 Priority": 0.45, "🟡 Standard": 0.40, "🟢 Passive": 0.15},
        }

        scenario_details = {
            "🟡  Balanced Approach": {
                "description": "Mid-range spend ($3K–$7K) blending short-term incentives with long-term engagement tactics. Suitable when churn risk is moderate and you want broad "
                "coverage without overspending.",
                "allocations": {
                    "Targeted Discount Offers":              (0.20, "Personalised discounts based on customer tenure and usage patterns."),
                    "Loyalty Rewards Program":               (0.20, "Points-based rewards redeemable for bill credits or service upgrades."),
                    "Personalised Email/SMS Campaigns":      (0.18, "Tailored messaging highlighting value and exclusive renewal offers."),
                    "Contract Upgrade Incentives":           (0.17, "Incentivise month-to-month customers to switch to annual plans."),
                    "Customer Success Check-ins":            (0.15, "Scheduled touchpoints to address satisfaction and resolve concerns."),
                    "Referral Bonuses":                      (0.10, "Reward retained customers for referring new subscribers."),
                }
            },
            "🟢  Cost-Efficient": {
                "description": "Low spend ($1K–$4K) using automation-first tactics to maximise reach per dollar. Best when budget is tight but the at-risk customer base is large — "
                "scale over personalisation.",
                "allocations": {
                    "Automated Email Retention Campaigns":   (0.30, "Behaviour-triggered emails at key churn-risk moments (e.g., 30-day inactivity)."),
                    "Self-Service Portal Improvements":      (0.25, "Invest in UX improvements to reduce friction and support costs."),
                    "Bundled Plan Upsells":                  (0.20, "Offer cost-saving bundles to reduce perceived price-to-value gap."),
                    "Community & Loyalty Program":           (0.15, "Build brand stickiness through community engagement and milestone rewards."),
                    "Targeted SMS Nudges":                   (0.10, "Short, timely SMS messages to re-engage at-risk customers cheaply."),
                }
            },
            "🔵  High-Value Customer Focus": {
                "description": "High spend ($5K–$10K) concentrated on a small segment of top CLV customers. Justified when losing one high-value customer costs more than "
                "the entire retention budget.",
                "allocations": {
                    "Dedicated Account Manager":             (0.30, "Assign a personal account manager to customers in the top CLV quartile."),
                    "VIP Loyalty Perks":                     (0.25, "Exclusive benefits such as priority support, early access, and partner discounts."),
                    "Exclusive Rate Lock Offers":            (0.20, "Guarantee current pricing for 12–24 months as a loyalty reward."),
                    "Premium Support SLA":                   (0.15, "Upgrade support tier to guaranteed response times and senior agents."),
                    "Personalised Retention Gifts":          (0.10, "Tangible goodwill gestures (e.g., gift cards, merchandise) for long-tenure VIPs."),
                }
            },
            "🤖  Automation-First": {
                "description": "Purely digital, zero manual-touch tactics. Scales to any budget — spend more to run more campaigns and A/B tests, spend less to maintain "
                "baseline automation. No headcount required.",
                "allocations": {
                    "Behavioural Trigger Emails":            (0.30, "Automated emails firing on specific signals: inactivity, bill spike, plan downgrade."),
                    "In-App / Portal Nudges":                (0.25, "Contextual messages inside the self-service portal surfacing upgrade or loyalty offers."),
                    "SMS Retention Sequences":               (0.20, "Multi-step SMS flows based on churn probability score, fully templated."),
                    "Chatbot Retention Flows":               (0.15, "AI chatbot intercepts cancellation intent and routes to a retention offer."),
                    "A/B Testing Budget":                    (0.10, "Continuous split-testing of messaging, offers, and timing to optimise conversion."),
                }
            },
        }

        # ── Scenario selector + budget slider ─────────w───────────────────────
        col_scenario, col_budget = st.columns(2)
        with col_scenario:
            scenario = st.selectbox(
                "Select Retention Scenario",
                list(scenario_budget_ranges.keys()),
            )
        with col_budget:
            st.markdown(f"### 💰 Budget: ${budget:,}")

            bmin, bmax, _ = scenario_budget_ranges[scenario]

            if budget < bmin or budget > bmax:
                st.warning(
                    f"⚠️ Outside recommended range (${bmin:,} – ${bmax:,}) "
                    f"for this scenario. Allocations will still be tailored to your ${budget:,} budget."
                )
            else:
                st.caption(f"💡 Recommended range: ${bmin:,} – ${bmax:,}")

        selected     = scenario_details[scenario]
        tier_weights = scenario_tier_weights[scenario]

        st.info(f"**Scenario Overview:** {selected['description']}")

        # ── Section A: Customer Intelligence by Tier ─────────────────────────
        st.subheader("📊 Customer Intelligence by Retention Tier")
        st.caption(
            "Customers are classified by **Churn Probability** (≥ 60% = high risk) and "
            "**CLV** (above/below dataset median). "
            "**Priority** = high risk AND high value — the highest ROI targets for any retention spend."
        )

        tier_rows = []
        for tier in ["🔴 Priority", "🟡 Standard", "🟢 Passive"]:
            tier_data      = data[data["RetentionTier"] == tier]
            n              = len(tier_data)
            var_sum        = tier_data["ValueAtRisk"].sum()
            tier_budget    = budget * tier_weights[tier]
            cost_per_cust  = tier_budget / n if n > 0 else 0
            avg_churn_prob = tier_data["ChurnProbability"].mean() if n > 0 else 0
            avg_clv        = tier_data["CLV_Calc"].mean() if n > 0 else 0
            tier_rows.append({
                "Tier":                    tier,
                "Customers":               n,
                "Avg Churn Prob":          f"{avg_churn_prob:.1%}",
                "Avg CLV ($)":             f"${avg_clv:,.0f}",
                "Total Value at Risk ($)": round(var_sum, 0),
                "Budget Weight":           f"{tier_weights[tier]*100:.0f}%",
                "Budget Allocated ($)":    round(tier_budget, 0),
                "Cost per Customer ($)":   round(cost_per_cust, 2),
            })

        tier_df = pd.DataFrame(tier_rows)
        st.dataframe(tier_df, width="stretch", hide_index=True)

        # Tier budget bar chart
        fig_tier = px.bar(
            tier_df,
            x="Tier",
            y="Budget Allocated ($)",
            color="Tier",
            text=tier_df["Budget Allocated ($)"].apply(lambda v: f"${v:,.0f}"),
            color_discrete_map={
                "🔴 Priority": "#EF553B",
                "🟡 Standard": "#FFA15A",
                "🟢 Passive":  "#00CC96",
            },
            title=f"Budget Allocated by Customer Tier — {scenario}",
        )
        fig_tier.update_traces(textposition="outside", cliponaxis=False)
        fig_tier.update_layout(
            showlegend=False,
            height=350,
            yaxis=dict(tickprefix="$", tickformat=","),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_tier, width="stretch")

        # ── Section B: Top Priority Customers ────────────────────────────────
        priority_customers = (
            data[data["RetentionTier"] == "🔴 Priority"]
            .sort_values("ValueAtRisk", ascending=False)
        )
        n_priority = len(priority_customers)

        st.subheader(f"🎯 Priority Customer List ({n_priority} customers)")
        st.caption(
            "Sorted by **Value at Risk** (Churn Probability × CLV). "
            "These are the customers where every dollar of retention spend delivers the highest expected return."
        )

        if n_priority > 0:
            slider_max = min(50, n_priority)
            slider_val = min(10, n_priority)
            top_n_p = st.slider(
                "Show top N priority customers",
                min_value=5,
                max_value=slider_max,
                value=slider_val,
                step=5,
                key="priority_slider",
            )
            display_cols = ["customerID", "ChurnProbability", "CLV_Calc", "ValueAtRisk", "RiskSegment", "MonthlyCharges", "tenure"]
            rename_map   = {"CLV_Calc": "CLV ($)", "ValueAtRisk": "Value at Risk ($)"}
            st.dataframe(
                priority_customers.head(top_n_p)[display_cols].rename(columns=rename_map),
                width="stretch",
                hide_index=True,
            )

            # Value at Risk bar chart for top N priority customers
            top_p_chart = priority_customers.head(top_n_p).copy()
            fig_var = px.bar(
                top_p_chart,
                x="customerID",
                y="ValueAtRisk",
                color="RiskSegment",
                text=top_p_chart["ValueAtRisk"].apply(lambda v: f"${v:,.0f}"),
                color_discrete_map={
                    "🔴 High Risk":    "#EF553B",
                    "🟡 Medium Risk":  "#FFA15A",
                    "🟢 Low Risk":     "#00CC96",
                },
                labels={"ValueAtRisk": "Value at Risk ($)", "customerID": "Customer ID"},
                title=f"Top {top_n_p} Priority Customers by Value at Risk",
                hover_data={"ChurnProbability": ":.1%", "CLV_Calc": ":,.0f"},
            )
            fig_var.update_traces(textposition="outside", cliponaxis=False)
            fig_var.update_layout(
                height=420,
                xaxis_tickangle=-45,
                yaxis=dict(tickprefix="$", tickformat=","),
                legend_title="Risk Segment",
                margin=dict(t=50, b=80),
            )
            st.plotly_chart(fig_var, width="stretch")
        else:
            st.info("No Priority-tier customers found in this dataset with the current thresholds.")

        # ── Section C: Strategy Budget Breakdown ─────────────────────────────
        st.subheader("💡 Strategy Budget Breakdown")
        priority_budget = budget * tier_weights["🔴 Priority"]
        st.caption(
            f"Strategy allocations below apply to the **Priority tier budget of ${priority_budget:,.0f}** "
            f"({tier_weights['🔴 Priority']*100:.0f}% of total ${budget:,.0f}). "
            "The remaining budget is split across Standard and Passive tiers as shown above."
        )

        strat_rows = []
        for strategy, (pct, description) in selected["allocations"].items():
            strat_rows.append({
                "Strategy":              strategy,
                "Allocated Budget ($)":  round(pct * priority_budget, 2),
                "Allocation (%)":        f"{int(pct * 100)}%",
                "Description":           description,
            })
        strategy_df = pd.DataFrame(strat_rows)

        st.dataframe(
            strategy_df[["Strategy", "Allocation (%)", "Allocated Budget ($)", "Description"]],
            width="stretch",
            hide_index=True,
        )

        chart_type = st.radio(
            "Chart View",
            ["Horizontal Bar", "Donut"],
            horizontal=True,
        )

        if chart_type == "Horizontal Bar":
            sorted_df = strategy_df.sort_values("Allocated Budget ($)")
            fig = px.bar(
                sorted_df,
                x="Allocated Budget ($)",
                y="Strategy",
                orientation="h",
                color="Strategy",
                text=sorted_df["Allocated Budget ($)"].apply(lambda v: f"${v:,.0f}"),
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Strategy Allocation (Priority Tier) — {scenario}",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                showlegend=False,
                xaxis_title="Allocated Budget ($)",
                yaxis_title="",
                height=420,
                margin=dict(l=10, r=80, t=50, b=40),
                xaxis=dict(tickprefix="$", tickformat=","),
            )
            st.plotly_chart(fig, width="stretch")

        else:
            fig = px.pie(
                strategy_df,
                names="Strategy",
                values="Allocated Budget ($)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Strategy Allocation (Priority Tier) — {scenario}",
            )
            fig.update_traces(
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>",
            )
            fig.update_layout(
                height=450,
                margin=dict(t=60, b=20),
                legend=dict(orientation="v", x=1.02, y=0.5),
            )
            st.plotly_chart(fig, width="stretch")

        # ── Section D: Expected ROI Calculator ───────────────────────────────
        st.subheader("📈 Expected ROI Calculator")
        st.caption(
            "Adjust the assumed success rate to model different campaign outcomes. "
            "**Expected CLV Saved** = Priority Tier Value at Risk × Success Rate."
        )

        success_rate = st.slider(
            "Assumed Retention Success Rate (%)",
            min_value=5,
            max_value=60,
            value=25,
            step=5,
            key="roi_slider",
            help="What % of Priority-tier customers do you expect to successfully retain?",
        )

        priority_var       = data[data["RetentionTier"] == "🔴 Priority"]["ValueAtRisk"].sum()
        expected_clv_saved = priority_var * (success_rate / 100)
        net_benefit        = expected_clv_saved - budget
        roi_pct            = (net_benefit / budget * 100) if budget > 0 else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Priority CLV at Risk",  f"${priority_var:,.0f}")
        r2.metric("Expected CLV Saved",    f"${expected_clv_saved:,.0f}",
                  f"@ {success_rate}% success rate")
        r3.metric("Total Budget Spent",    f"${budget:,.0f}")
        r4.metric(
            "Net Benefit",
            f"${net_benefit:,.0f}",
            f"ROI: {roi_pct:.0f}%",
            delta_color="normal" if net_benefit >= 0 else "inverse",
        )

        if net_benefit >= 0:
            st.success(
                f"✅ At a **{success_rate}% success rate**, retaining Priority customers is projected to return "
                f"**\\${expected_clv_saved:,.0f}** in preserved CLV against a **\\${budget:,.0f}** spend — "
                f"a net benefit of **${net_benefit:,.0f}** ({roi_pct:.0f}% ROI)."
            )
        else:
            st.warning(
                f"⚠️ At a **{success_rate}% success rate**, the campaign does not cover its cost. "
                f"Consider increasing the success rate assumption, reducing the budget, or focusing on a higher-CLV segment."
            )

        st.caption(
            "⚠️ This is a financial projection, not a guarantee. "
            "Expected CLV Saved assumes the retained customers maintain their current monthly spend for the remainder of their modelled tenure."
        )

# ----------------------------------------
# 6. Evaluation Panel
# ----------------------------------------
        st.header("Model Evaluation")
        st.info(f"📊 Model in Use: **{model_choice}**")

        if model_choice == "Logistic Regression":
            cv_scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
            )
            st.caption(
                f"📊 Cross-validated AUC (5-fold): "
                f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f} — "
                f"more reliable than single split AUC above."
            )

        y_pred       = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)  # ← only this line changes
        y_pred_proba = model.predict_proba(X_test)[:, 1]                         # ← untouched

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_pred_proba)

        baseline = DummyClassifier(strategy="most_frequent", random_state=42)
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)

        baseline_acc  = accuracy_score(y_test, baseline_pred)
        baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
        baseline_rec  = recall_score(y_test, baseline_pred, zero_division=0)
        baseline_f1   = f1_score(y_test, baseline_pred, zero_division=0)

        with st.expander("Baseline Comparison", expanded=False):
            st.caption(
                "Comparing the trained model against a naive baseline that always predicts "
                "the majority class, to confirm the model adds genuine predictive value. "
                "AUC-ROC is excluded for the baseline as it produces no probability estimates."
            )

            base_cols = st.columns(4)
            base_cols[0].metric(
                "Baseline Accuracy",
                f"{baseline_acc:.1%}",
                f"{(acc - baseline_acc)*100:+.1f} pts vs model",
            )
            base_cols[1].metric(
                "Baseline Precision",
                f"{baseline_prec:.1%}",
                f"{(prec - baseline_prec)*100:+.1f} pts vs model",
            )
            base_cols[2].metric(
                "Baseline Recall",
                f"{baseline_rec:.1%}",
                f"{(rec - baseline_rec)*100:+.1f} pts vs model",
            )
            base_cols[3].metric(
                "Baseline F1",
                f"{baseline_f1:.1%}",
                f"{(f1 - baseline_f1)*100:+.1f} pts vs model",
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{acc:.1%}")
        m2.metric("Precision", f"{prec:.1%}")
        m3.metric("Recall",    f"{rec:.1%}")
        m4.metric("F1 Score",  f"{f1:.1%}")
        m5.metric("AUC-ROC",   f"{auc:.3f}")

        st.caption(
            "ℹ️ **Precision** = of predicted churners, how many truly churn. "
            "**Recall** = of true churners, how many we caught. "
            "**AUC-ROC** closer to 1.0 = stronger discrimination."
        )

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                height=380,
                margin=dict(t=50, b=40, l=10, r=10),
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_cm, width="stretch")

        with eval_col2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"Random Forest (AUC = {auc:.3f})",
                line=dict(color="#636EFA", width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Baseline",
                line=dict(color="grey", dash="dash"),
            ))
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=380,
                margin=dict(t=50, b=40, l=10, r=10),
                legend=dict(x=0.55, y=0.05),
            )
            st.plotly_chart(fig_roc, width="stretch")

        # ── Feature Importance ─────────────────────────────────────────────
        st.subheader("Feature Importance")

        if model_choice in ["Random Forest"]:
            importance_values = model.feature_importances_
            importance_label  = "Importance Score"

        else:  # Logistic Regression
            importance_values = np.abs(model.coef_[0])
            importance_label  = "Coefficient Magnitude"

        importances = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importance_values,
        }).sort_values("Importance", ascending=True).tail(15)

        importances["Feature"] = importances["Feature"].str.replace("_enc$", " (cat)", regex=True)

        fig_imp = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title=f"Top 15 Feature Importances ({model_choice})",
            labels={"Importance": importance_label, "Feature": ""},
        )

        fig_imp.update_layout(
            height=460,
            coloraxis_showscale=False,
            margin=dict(l=10, r=40, t=50, b=40),
        )

        st.plotly_chart(fig_imp, width="stretch")
        st.caption("Features with higher importance scores have more influence on the model's churn predictions.")
