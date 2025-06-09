# app.py

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import warnings
import traceback

warnings.filterwarnings('ignore')

# Modern time series and machine learning libraries
try:
    from pmdarima import auto_arima
    from pmdarima.arima import ADFTest
    PMDARIMA_AVAILABLE = True
except ImportError as e:
    PMDARIMA_AVAILABLE = False
    PMDARIMA_ERROR = str(e)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Detect LightGBM availability and capture any import error
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = e

# Fallback to statsmodels if needed
import statsmodels.api as sm

# at the very top, after imports
forecast_results = None
fallback_reason  = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Set wide layout for better screen utilization
st.set_page_config(
    page_title="MBS CPR Forecaster",
    page_icon="images/Logo-37.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure Plotly theme
plotly_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans serif", color="#111111"),
        colorway=["#667eea", "#20c997"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=80, b=80),
        height=600
    )
)
pio.templates["mbs_theme"] = plotly_template
pio.templates.default = "mbs_theme"

# Database connection
# Note: Update this path to your database file
con = duckdb.connect('/home/gregoliven/data2/mbs/mbs.db', read_only=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODELING FUNCTIONS - TEMPORAL vs CROSS-SECTIONAL APPROACHES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_model_type(cpr_series, external_features=None):
    """
    Analyze data characteristics to determine optimal modeling approach.
    """
    n_obs = len(cpr_series)
    cpr_std = cpr_series.std()
    cpr_mean = cpr_series.mean()
    cv = cpr_std / cpr_mean if cpr_mean > 0 else 0
    
    autocorr_1 = cpr_series.autocorr(lag=1) if n_obs > 1 else 0
    autocorr_3 = cpr_series.autocorr(lag=3) if n_obs > 3 else 0
    has_strong_temporal = abs(autocorr_1) > 0.3 or abs(autocorr_3) > 0.2
    
    has_variation = cv > 0.1 and cpr_std > 0.5
    sufficient_length = n_obs >= 24
    has_external = external_features is not None and len(external_features.columns) > 1
    
    if has_external and sufficient_length and has_variation and has_strong_temporal:
        return "multivariate", f"Strong temporal patterns (ACF: {autocorr_1:.2f}) and rich features."
    elif has_external and sufficient_length and has_variation:
        return "gradient_boosting", "Rich features available for cross-sectional modeling."
    elif sufficient_length and has_variation and has_strong_temporal:
        return "univariate_complex", f"Strong temporal patterns (ACF: {autocorr_1:.2f})."
    elif has_variation:
        return "univariate_simple", "Limited data or weak patterns."
    else:
        return "persistence", "Data appears stable/flat."

def prepare_seller_level_features(seller_data, target_col='cpr'):
    """
    Feature engineering using seller-level aggregated data.
    This prepares data for LightGBM training and forecasting.
    """
    df = seller_data.copy()
    
    # Temporal features from the index (month and year are already in SQL query if needed)
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Select features - these are columns already calculated in the SQL query or derived
    feature_cols = [
        'weighted_avg_rate', 'weighted_avg_ltv', 'weighted_avg_dti', 'weighted_avg_credit_score',
        'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', 'cpr_6m_avg',
        'cpr_1m_lag', 'cpr_3m_lag',
        'refi_incentive', 'rate_volatility',
        'month_sin', 'month_cos', 'time_index',
        'pmms30_trend'
    ]
    
    # Ensure all selected feature columns exist in the dataframe, use only available ones
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Remove rows with missing target or features
    df = df.dropna(subset=[target_col] + available_feature_cols)
    
    X = df[available_feature_cols]
    y = df[target_col] * 100  # Convert CPR to percentage
    
    return X, y, available_feature_cols

def fit_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    Train a Gradient Boosting model (LightGBM) for CPR prediction.
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 1000  # Will be controlled by early stopping
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model

# def forecast_with_lightgbm(model, seller_data, horizon, feature_names):
#     """
#     Generate forecasts using the trained LightGBM model.
#     This function iteratively predicts one step at a time.
#     """
#     # Start with the last known observation from the seller data
#     last_obs = seller_data.iloc[-1:].copy()
    
#     # Verify that required columns exist
#     required_cols = ['time_index', 'pmms30_trend']  # Add pmms30_trend to required columns
#     missing_cols = [col for col in required_cols if col not in last_obs.columns]
#     if missing_cols:
#         raise ValueError(f"Columns {missing_cols} not found in seller_data. Ensure they are included in the SQL query.")
    
#     forecasts = []
    
#     for i in range(horizon):
#         # Prepare the feature set for the next period
#         future_features = last_obs.copy()
#         future_features.index = last_obs.index + pd.DateOffset(months=1)
        
#         # Update time-based features
#         future_features['month'] = future_features.index.month
#         future_features['year'] = future_features.index.year
#         current_time_index = future_features['time_index'].iloc[0]  # Extract scalar value
#         future_features['time_index'] = current_time_index + 1     # Increment and assign
#         future_features['month_sin'] = np.sin(2 * np.pi * future_features['month'] / 12)
#         future_features['month_cos'] = np.cos(2 * np.pi * future_features['month'] / 12)

#         # Update lagged CPR features with the previous prediction
#         if i == 0:
#             future_features['cpr_1m_lag'] = last_obs['cpr'].iloc[0]
#             future_features['cpr_3m_lag'] = seller_data['cpr'].iloc[-3] if len(seller_data) >= 3 else last_obs['cpr'].iloc[0]
#         else:
#             future_features['cpr_1m_lag'] = forecasts[-1] / 100.0  # Convert back to ratio
#             if i >= 3:
#                 future_features['cpr_3m_lag'] = forecasts[-3] / 100.0
#             else:
#                 future_features['cpr_3m_lag'] = seller_data['cpr'].iloc[-3+i]

#         # Keep pmms30_trend constant (persistence) for now
#         # Other external features (like pmms30) also remain constant
        
#         # Ensure the feature vector has the same columns in the same order as the training data
#         X_step = future_features[feature_names]
        
#         # Generate prediction
#         prediction = model.predict(X_step)[0]
#         prediction = max(0.1, min(50.0, prediction))  # Bound the prediction
#         forecasts.append(prediction)
        
#         # Update last_obs for the next iteration
#         last_obs = future_features
#         last_obs['cpr'] = prediction / 100.0  # Update CPR with the new prediction

#     return np.array(forecasts)
def forecast_with_lightgbm(model, seller_data, horizon, feature_names):
    """
    Generate forecasts using the trained LightGBM model.
    This function iteratively predicts one step at a time.
    """
    # Start with the last known observation from the seller data
    last_obs = seller_data.iloc[-1:].copy()
    
    # Verify that required columns exist
    required_cols = ['time_index', 'pmms30_trend']
    missing_cols = [col for col in required_cols if col not in last_obs.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in seller_data. Ensure they are included in the SQL query.")
    
    forecasts = []
    last_trend_values = seller_data['pmms30_trend'].tail(3).values  # Last 3 months for trend
    trend_slope = np.mean(np.diff(last_trend_values)) if len(last_trend_values) > 1 else 0
    
    for i in range(horizon):
        # Prepare the feature set for the next period
        future_features = last_obs.copy()
        future_features.index = last_obs.index + pd.DateOffset(months=1)
        
        # Update time-based features
        future_features['month'] = future_features.index.month
        future_features['year'] = future_features.index.year
        current_time_index = future_features['time_index'].iloc[0]
        future_features['time_index'] = current_time_index + 1
        future_features['month_sin'] = np.sin(2 * np.pi * future_features['month'] / 12)
        future_features['month_cos'] = np.cos(2 * np.pi * future_features['month'] / 12)

        # Update lagged CPR features with the previous prediction
        if i == 0:
            future_features['cpr_1m_lag'] = last_obs['cpr'].iloc[0]
            future_features['cpr_3m_lag'] = seller_data['cpr'].iloc[-3] if len(seller_data) >= 3 else last_obs['cpr'].iloc[0]
        else:
            future_features['cpr_1m_lag'] = forecasts[-1] / 100.0
            if i >= 3:
                future_features['cpr_3m_lag'] = forecasts[-3] / 100.0
            else:
                future_features['cpr_3m_lag'] = seller_data['cpr'].iloc[-3+i]

        # Propagate pmms30_trend based on historical trend
        future_features['pmms30_trend'] = last_obs['pmms30_trend'].iloc[0] + trend_slope * (i + 1)
        
        # Ensure the feature vector has the same columns in the same order as the training data
        X_step = future_features[feature_names]
        
        # Generate prediction
        prediction = model.predict(X_step)[0]
        prediction = max(0.1, min(50.0, prediction))
        forecasts.append(prediction)
        
        # Update last_obs for the next iteration
        last_obs = future_features
        last_obs['cpr'] = prediction / 100.0

    return np.array(forecasts)

def fit_prophet_model(cpr_data, external_features=None):
    """
    Fit Facebook Prophet model for robust time series forecasting.
    """
    try:
        prophet_df = pd.DataFrame({
            'ds': cpr_data.index,
            'y': cpr_data.values
        })
        
        # --- START OF FIX: Fine-tuning Prophet's hyperparameters for stability ---
        model = Prophet(
            seasonality_mode='additive',      # More stable for series that don't have huge swings
            changepoint_prior_scale=0.05,     # Makes the trend more stable, less prone to overfitting
            seasonality_prior_scale=5.0,      # Reduces the magnitude of the seasonal swings
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        if external_features is not None:
            # Add regressors to Prophet model
            for col in external_features.columns:
                if col != 'cpr': # Target should not be a regressor
                    model.add_regressor(col)
                    prophet_df[col] = external_features[col].values
        
        model.fit(prophet_df)
        return model, prophet_df, True
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)[:100]}...")
        return None, None, False

def create_simple_forecast(cpr_series, horizon):
    """
    Simple persistence/trend model for problematic data.
    """
    recent_values = cpr_series.tail(6)
    mean_cpr = recent_values.mean()
    std_cpr = recent_values.std()
    
    # Use a simple linear trend on recent data
    x = np.arange(len(recent_values))
    trend = np.polyfit(x, recent_values.values, 1)[0] if len(recent_values) > 1 else 0
    
    forecast_values = []
    for i in range(horizon):
        forecast_val = mean_cpr + (trend * (i + 1))
        forecast_val = max(0, min(100, forecast_val)) # Bound the forecast
        forecast_values.append(forecast_val)
    
    margin = max(std_cpr * 1.96, 0.5)
    lower_ci = [max(0, f - margin) for f in forecast_values]
    upper_ci = [min(100, f + margin) for f in forecast_values]
    
    return np.array(forecast_values), np.array([lower_ci, upper_ci]).T

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE HEADER & USER CONTROLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div style="margin: 0 20px;">', unsafe_allow_html=True)

header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.image("images/Logo-37.png", width=80)
with header_col2:
    st.title("MBS CPR Time Series Forecaster")
    st.caption("Powered by Polygon Research")

st.markdown("---")

with st.expander("📊 Model Configuration", expanded=True):
    config_col1, config_col2, config_col3, config_col4 = st.columns([3, 2, 2, 1])
    
    with config_col1:
        sellers_query = """
            SELECT DISTINCT
                CASE 
                    WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' THEN 'UNITED WHOLESALE MORTGAGE, LLC' 
                    ELSE seller_name 
                END as seller
            FROM main.gse_sf_mbs 
            WHERE is_in_bcpr3 AND prefix = 'CL' 
            AND as_of_month >= date_trunc('month', now()) - interval '6 months'
            GROUP BY 1
            HAVING COUNT(*) >= 1000
            ORDER BY COUNT(*) DESC
            LIMIT 50;
        """
        try:
            sellers = [row[0] for row in con.execute(sellers_query).fetchall()]
            # Ensure UWM is always an option and at the top
            uwm_name = 'UNITED WHOLESALE MORTGAGE, LLC'
            if uwm_name in sellers:
                sellers.remove(uwm_name)
            sellers.insert(0, uwm_name)
            selected_seller = st.selectbox("Select Seller", sellers, index=0)
        except Exception as e:
            st.error(f"Could not load seller list from database: {e}")
            st.stop()
    
    with config_col2:
        forecast_horizon = st.number_input("Forecast Horizon (months)", value=6, min_value=1, max_value=24)
    
    with config_col3:
        model_type = st.selectbox("Model Type", 
                                 ["Auto-Select", "LightGBM (Gradient Boosting)", "Prophet (Time Series)", "Auto-ARIMA (Classical)", "Simple (Baseline)"],
                                 help="Choose modeling paradigm.")
    
    with config_col4:
        st.write("")
        run_prediction = st.button("🚀 **Run Forecast**", type="primary", use_container_width=True)

# FIX: Properly escape the seller name for use in SQL queries
safe_seller_name = selected_seller.replace("'", "''")

# Define the seller name logic for the WHERE clause
seller_where_clause = f"""
    (seller_name = '{safe_seller_name}' 
     OR ('{safe_seller_name}' = 'UNITED WHOLESALE MORTGAGE, LLC' AND seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC'))
"""

seller_info_query = f"""
    SELECT 
        COUNT(*),
        COUNT(DISTINCT DATE_TRUNC('month', as_of_month)),
        MIN(as_of_month),
        MAX(as_of_month)
    FROM main.gse_sf_mbs 
    WHERE is_in_bcpr3 AND prefix = 'CL' AND {seller_where_clause} AND loan_correction_indicator != 'pri'
    AND as_of_month >= '2022-01-01';
"""
seller_info = con.execute(seller_info_query).fetchone()
st.info(f"📊 **{selected_seller}**: {seller_info[0]:,} total loan observations across {seller_info[1]} months "
        f"({pd.to_datetime(seller_info[2]).strftime('%b %Y')} - {pd.to_datetime(seller_info[3]).strftime('%b %Y')})")

if not run_prediction:
    st.info("👆 **Select your configuration and click 'Run Forecast' to begin.**")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & DUAL QUERY APPROACH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Loading and preparing data..."):
    
    # Query 1: Loan-level data (only if needed for diagnostics, not for modeling)
    # CASE WHEN prepayable_balance > 0 THEN 1 - POWER(1 - (unscheduled_principal_payment / prepayable_balance), 12) ELSE 0 END as cpr,
    loan_level_query = f"""
        SELECT 
            loan_identifier as loan_id,
            DATE_TRUNC('month', as_of_month) as month,
            cpr,
            current_interest_rate_pri as loan_rate,
            ltv, credit_score,
            loan_rate - COALESCE(pmms30, 0) as refi_incentive,
            CASE WHEN COALESCE(months_delinquent, 0) > 0 THEN 1 ELSE 0 END as is_delinquent
        FROM main.gse_sf_mbs a 
        LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
        WHERE is_in_bcpr3 AND prefix = 'CL' 
        AND {seller_where_clause}
        AND as_of_month >= '2022-01-01'
        AND loan_correction_indicator != 'pri' AND prepayable_balance > 0
        AND current_interest_rate_pri IS NOT NULL AND ltv IS NOT NULL AND credit_score IS NOT NULL
    """
    
    # Query 2: Seller-level aggregated data for all time series models
    seller_level_query = f"""
        WITH seller_monthly AS (
                SELECT 
                    DATE_TRUNC('month', as_of_month) as month,
                    SUM(1) as loan_count,
                    SUM(current_investor_loan_upb) as total_upb,
                    SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
                    SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
                    SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
                    SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
                    CASE WHEN SUM(prepayable_balance) > 0 
                        THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) 
                        ELSE 0 END as cpr,
                    AVG(pmms30) as pmms30,
                    AVG(pmms30_1m_lag) as pmms30_1m_lag,
                    AVG(pmms30_2m_lag) as pmms30_2m_lag,
                    ROW_NUMBER() OVER (ORDER BY DATE_TRUNC('month', as_of_month)) - 1 as time_index,
                    SIN(2 * PI() * EXTRACT(MONTH FROM DATE_TRUNC('month', as_of_month)) / 12) as month_sin,
                    COS(2 * PI() * EXTRACT(MONTH FROM DATE_TRUNC('month', as_of_month)) / 12) as month_cos
                FROM main.gse_sf_mbs a 
                LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
                WHERE is_in_bcpr3 AND prefix = 'CL' AND {seller_where_clause}
                AND as_of_month >= '2022-01-01' AND loan_correction_indicator != 'pri'
                GROUP BY DATE_TRUNC('month', as_of_month)
            )
            SELECT *,
                LAG(cpr, 1) OVER (ORDER BY month) as cpr_1m_lag,
                LAG(cpr, 3) OVER (ORDER BY month) as cpr_3m_lag,
                AVG(cpr) OVER (ORDER BY month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as cpr_6m_avg,
                AVG(pmms30) OVER (ORDER BY month ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as pmms30_trend,
                weighted_avg_rate - pmms30 as refi_incentive,
                ABS(pmms30 - pmms30_1m_lag) as rate_volatility
            FROM seller_monthly
            ORDER BY month
    """
    
    loan_data = None
    if model_type == "LightGBM" or model_type == "Auto-Select":
        st.info("📊 Loading loan-level data for diagnostics...")
        loan_data = con.execute(loan_level_query).df()
        if len(loan_data) > 0:
            st.success(f"✅ Loaded {len(loan_data):,} loan-month observations for diagnostics.")
        else:
            st.warning("⚠️ No loan-level data found for this period.")

    st.info("📊 Loading seller-level aggregate data for modeling...")
    seller_data = con.execute(seller_level_query).df()
    st.write(f"Debug: Columns in seller_data: {seller_data.columns.tolist()}")
    
    if len(seller_data) > 0:
        seller_data['month'] = pd.to_datetime(seller_data['month'])
        seller_data = seller_data.set_index('month').sort_index()
        
        # FIX: Drop rows with NaN values created by SQL LAG functions.
        # This ensures all models receive clean, complete data.
        initial_rows = len(seller_data)
        seller_data = seller_data.dropna()
        final_rows = len(seller_data)
        
        if final_rows > 0:
            st.success(f"✅ Prepared {final_rows} months of clean seller-level data for modeling.")
            if initial_rows > final_rows:
                st.info(f"ℹ️ Dropped {initial_rows - final_rows} initial row(s) that had missing lag data.")
        else:
            st.error("❌ After cleaning, no data remained for modeling. The time series may be too short.")
            st.stop()
    else:
        st.error(f"❌ No seller-level data found for {selected_seller}. Cannot proceed.")
        st.stop()

# Prepare time series for analysis
cpr_series = seller_data['cpr'] * 100

# Prepare feature set for Prophet/LightGBM
external_features = seller_data.drop(columns=['cpr', 'loan_count', 'total_upb']).copy()

# Data Preview Expander
with st.expander("📊 Data Preview"):
    st.write("**Seller-Level Aggregated Data (used for modeling):**")
    st.dataframe(seller_data.head())
    if loan_data is not None:
        st.write("**Loan-Level Sample Data (for diagnostics):**")
        st.dataframe(loan_data.head())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING & FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Training models and generating forecasts..."):
    
    if model_type == "Auto-Select":
        suggested_model, reasoning = detect_model_type(cpr_series, external_features)
        st.info(f"🤖 **Auto-Selected Model: {suggested_model.replace('_', ' ').title()}** ({reasoning})")
    else:
        model_map = { "LightGBM (Gradient Boosting)": "gradient_boosting", "Prophet (Time Series)": "multivariate", "Auto-ARIMA (Classical)": "univariate_complex", "Simple (Baseline)": "persistence"}
        suggested_model = model_map.get(model_type, "persistence")

    forecast_results = {}
    model_info = {}

    fallback_reason = None

    # initialize a variable to capture *why* we ended up here
    fallback_reason = None
    
    # Model 1: LightGBM
    if LIGHTGBM_AVAILABLE and suggested_model == "gradient_boosting":
        st.write("🚀 **Training LightGBM Model** (Gradient Boosting)")
        try:
            X, y, feature_names = prepare_seller_level_features(seller_data, 'cpr')
            st.info(f"ℹ️ After preparing features for LightGBM, {len(X)} data points are available for training.")

            if len(X) > 12:
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.info(f"ℹ️ Training set size: {len(X_train)}, Test set size: {len(X_test)}")
                # Train LightGBM model with adjusted parameters
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,  # Increased to use more features
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbose': -1,
                    'n_estimators': 2000,
                    'max_depth': 10  # Added to control tree depth
                }
                lgb_model = lgb.LGBMRegressor(**params)
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                st.info("ℹ️ LightGBM model training completed.")
                # Generate forecasts
                forecast_values = forecast_with_lightgbm(lgb_model, seller_data, forecast_horizon, feature_names)
                st.info(f"ℹ️ Generated {len(forecast_values)} forecast values.")
                # Confidence interval
                residuals = y_test - lgb_model.predict(X_test)
                margin = np.std(residuals) * 1.96  # 95% CI
                conf_int = np.array([[max(0, v - margin * (1 + i * 0.1)) for i, v in enumerate(forecast_values)],
                                    [min(100, v + margin * (1 + i * 0.1)) for i, v in enumerate(forecast_values)]]).T
                forecast_results['LightGBM'] = {'values': forecast_values, 'confidence': conf_int}
                model_info['LightGBM'] = {
                    'type': 'LightGBM',
                    'paradigm': 'Gradient Boosting',
                    'score': f"RMSE: {np.sqrt(mean_squared_error(y_test, lgb_model.predict(X_test))):.2f}",
                    'features': feature_names
                }
                # Display feature importances
                importances = lgb_model.feature_importances_
                if len(importances) != len(feature_names):
                    st.warning("⚠️ Mismatch between feature importances and feature names. Check data alignment.")
                feature_importance_dict = {f: int(i) for f, i in zip(feature_names, importances)}
                st.write("**Top 5 Features by Importance:**")
                st.json(dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]))
                st.success("✅ LightGBM model trained and forecast generated.")
            else:
                fallback_reason = f"Insufficient data for LightGBM: only {len(X)} data points (need >12)."
                st.warning(f"⚠️ {fallback_reason} Falling back to Simple model.")
        except Exception as e:
            fallback_reason = f"Exception in LightGBM: {str(e)}\nTraceback: {traceback.format_exc()}"
            st.warning(f"⚠️ {fallback_reason} Falling back to Simple model.")

    # Model 2: Prophet
    elif PROPHET_AVAILABLE and suggested_model == "multivariate":
        # (The Prophet code from our previous fix goes here - no changes needed to it)
        prophet_regressors = [ 'refi_incentive', 'rate_volatility', 'weighted_avg_credit_score' ]
        final_prophet_regressors = [col for col in prophet_regressors if col in external_features.columns]
        st.write(f"🔮 **Training Prophet Model** with curated regressors: `{', '.join(final_prophet_regressors)}`")
        model, prophet_df, success = fit_prophet_model(cpr_series, external_features[final_prophet_regressors])
        if success:
            future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
            historical_regressors = external_features[final_prophet_regressors].reset_index().rename(columns={'month': 'ds'})
            future = pd.merge(future, historical_regressors, on='ds', how='left')
            future[final_prophet_regressors] = future[final_prophet_regressors].fillna(method='ffill')
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].tail(forecast_horizon).values
            forecast_values[forecast_values < 0] = 0
            forecast_results['Prophet'] = {'values': forecast_values, 'confidence': forecast[['yhat_lower', 'yhat_upper']].tail(forecast_horizon).values}
            model_info['Prophet'] = {'type': 'Prophet', 'paradigm': 'Time Series Decomposition', 'score': 'N/A', 'features': final_prophet_regressors}
            st.success("✅ Prophet model trained.")

    # Model 3: Auto-ARIMA
    elif PMDARIMA_AVAILABLE and "univariate" in suggested_model:
        st.write("📈 **Training Auto-ARIMA Model** (Classical Econometrics)")
        seasonal = len(cpr_series) >= 24
        model = auto_arima(cpr_series, seasonal=seasonal, m=12 if seasonal else 1, suppress_warnings=True, error_action='ignore', stepwise=True)
        forecast_values, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_results['Auto-ARIMA'] = {'values': forecast_values, 'confidence': conf_int}
        model_info['Auto-ARIMA'] = {'type': f'ARIMA{model.order}', 'paradigm': 'Econometric Time Series', 'score': f"{model.aic():.1f} AIC", 'features': ['Lagged CPR']}
        st.success(f"✅ Auto-ARIMA model trained: {model.order}")

    # Fallback: Simple Model
    if not forecast_results:  # Check if forecast_results is empty
        st.warning("🔄 Fallback to Simple model: No forecasts generated by selected model.")
        if not LIGHTGBM_AVAILABLE:
            st.error(f"❌ LightGBM unavailable: {LIGHTGBM_IMPORT_ERROR}")
        elif fallback_reason:
            st.info(f"ℹ️ Fallback reason: {fallback_reason}")
        else:
            st.info(f"ℹ️ Fallback reason: Suggested model ({suggested_model}) did not produce results.")

        st.write("📊 **Using Simple Baseline Model**")
        st.info("Using simple trend-based forecast as primary or fallback.")

        forecast_values, conf_int = create_simple_forecast(cpr_series, forecast_horizon)
        forecast_results['Simple'] = {'values': forecast_values, 'confidence': conf_int}
        model_info['Simple'] = {
            'type': 'Simple Trend',
            'paradigm': 'Statistical Baseline',
            'score': 'N/A',
            'features': ['Recent Trend']
        }

    if not forecast_results:
        st.error("❌ All models failed to produce a forecast. Cannot continue.")
        st.stop()
    
    primary_model = list(forecast_results.keys())[0]
    forecast_df = pd.DataFrame({
        'forecast': forecast_results[primary_model]['values'],
        'lower_CI': forecast_results[primary_model]['confidence'][:, 0],
        'upper_CI': forecast_results[primary_model]['confidence'][:, 1],
    }, index=pd.date_range(start=cpr_series.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq='MS'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS & RESULTS DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader(f"Forecast Results for {selected_seller}")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
current_cpr = cpr_series.iloc[-1]
next_month_cpr = forecast_df['forecast'].iloc[0]
delta = next_month_cpr - current_cpr
ci_width = forecast_df['upper_CI'].iloc[0] - forecast_df['lower_CI'].iloc[0]

kpi_col1.metric("Next-Month CPR", f"{next_month_cpr:.1f}%", f"{delta:+.1f}pp vs current")
kpi_col2.metric("Confidence Range (95%)", f"±{ci_width/2:.1f}pp", help="The 95% confidence interval for the next month's forecast.")
kpi_col3.metric("Current CPR", f"{current_cpr:.1f}%", help="The most recently observed CPR value.")
kpi_col4.metric("Active Model", primary_model, help=model_info[primary_model]['paradigm'])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORECAST VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig = go.Figure()
fig.add_trace(go.Scatter(x=cpr_series.index, y=cpr_series.values, mode='lines+markers', name='Historical CPR', line=dict(color='#667eea', width=3)))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines+markers', name=f'{primary_model} Forecast', line=dict(color='#20c997', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                         y=list(forecast_df['upper_CI']) + list(forecast_df['lower_CI'][::-1]),
                         fill='toself', fillcolor='rgba(32, 201, 151, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                         hoverinfo="skip", showlegend=False, name='95% CI'))
fig.update_layout(
    template="mbs_theme",
    title=dict(text=f"{forecast_horizon}-Month CPR Forecast for {selected_seller}", x=0.5, font=dict(size=24)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="CPR (%)"
)
st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DIAGNOSTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("🔧 Model Diagnostics & Feature Analysis"):
    info = model_info[primary_model]
    st.subheader(f"**{primary_model}** - {info['type']}")
    st.write(f"*Paradigm*: {info['paradigm']}")
    
    diag_cols = st.columns(3)
    diag_cols[0].metric("Performance Score", info['score'])
    diag_cols[1].metric("Features Used", len(info['features']))
    diag_cols[2].metric("Data Points", len(cpr_series))
    
    if 'LightGBM' in primary_model and 'features' in info:
        st.write("**Top 5 Features by Importance:**")
        st.json({f: int(v) for f, v in zip(info['features'], lgb_model.feature_importances_)})

st.markdown('</div>', unsafe_allow_html=True)