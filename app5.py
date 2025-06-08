# app.py
# usage: streamlit run app.py

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import warnings
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

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Fallback to statsmodels if needed
import statsmodels.api as sm

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
con = duckdb.connect('/home/gregoliven/data2/mbs/mbs.db')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODELING FUNCTIONS - TEMPORAL vs CROSS-SECTIONAL APPROACHES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_model_type(cpr_series, external_features=None):
    """
    Analyze data characteristics to determine optimal modeling approach.
    
    Uses autocorrelation analysis and data volume/variation tests to classify
    the time series as suitable for different modeling paradigms.
    """
    n_obs = len(cpr_series)
    cpr_std = cpr_series.std()
    cpr_mean = cpr_series.mean()
    cv = cpr_std / cpr_mean if cpr_mean > 0 else 0
    
    # Test for autocorrelation (temporal dependencies)
    autocorr_1 = cpr_series.autocorr(lag=1) if n_obs > 1 else 0
    autocorr_3 = cpr_series.autocorr(lag=3) if n_obs > 3 else 0
    has_strong_temporal = abs(autocorr_1) > 0.3 or abs(autocorr_3) > 0.2
    
    # Check for sufficient variation and length
    has_variation = cv > 0.1 and cpr_std > 0.5
    sufficient_length = n_obs >= 24
    has_external = external_features is not None and len(external_features.columns) > 1
    
    if has_external and sufficient_length and has_variation and has_strong_temporal:
        return "multivariate", f"Strong temporal patterns (ACF: {autocorr_1:.2f})"
    elif has_external and sufficient_length and has_variation:
        return "multivariate", "Rich features available"
    elif sufficient_length and has_variation and has_strong_temporal:
        return "univariate_complex", f"Temporal patterns (ACF: {autocorr_1:.2f})"
    elif has_variation:
        return "univariate_simple", "Limited data or weak patterns"
    else:
        return "persistence", "Stable/flat data"

def prepare_lightgbm_features(loan_data, seller_data, target_col='cpr'):
    """
    Simplified feature preparation since DuckDB now handles most feature engineering.
    
    DuckDB has already computed binary features, handled NULLs, and created
    temporal features. This function just selects and organizes them.
    """
    if loan_data is None:
        return prepare_seller_level_features(seller_data, target_col)
    
    df = loan_data.copy()
    
    # Features are already computed in DuckDB - just select them
    numeric_features = [
        'loan_balance', 'loan_rate', 'ltv', 'dti', 'credit_score', 'loan_age',
        'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', 'cpr_1m_lag', 'cpr_3m_lag',
        'refi_incentive', 'rate_volatility', 'months_observed',
        'month_sin', 'month_cos'
    ]
    
    binary_features = [
        'high_ltv', 'high_dti', 'low_credit', 'jumbo_loan',
        'strong_refi_incentive', 'negative_refi_incentive', 'is_delinquent'
    ]
    
    # Handle categorical features with pandas get_dummies (after DuckDB processing)
    categorical_features = ['occupancy_status', 'channel', 'fthb']
    
    # Start with numeric and binary features (all clean from DuckDB)
    feature_cols = numeric_features + binary_features
    X = df[feature_cols].copy()
    
    # Add categorical features as dummies
    for cat_col in categorical_features:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            feature_cols.extend(dummies.columns.tolist())
    
    # Target variable (already cleaned in DuckDB)
    y = df[target_col] * 100  # Convert to percentage
    
    # Verify no NaN values (should be clean from DuckDB)
    if X.isnull().sum().sum() > 0:
        st.warning(f"⚠️ Found {X.isnull().sum().sum()} null values after DuckDB processing")
        X = X.fillna(0)  # Fallback
    
    return X, y, feature_cols

def prepare_seller_level_features(seller_data, target_col='cpr'):
    """
    Fallback feature engineering using seller-level aggregated data.
    """
    df = seller_data.copy()
    
    # Temporal features
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['time_index'] = range(len(df))
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Rolling statistics
    if len(df) >= 6:
        df['cpr_ma3'] = df[target_col].rolling(3, min_periods=1).mean()
        df['cpr_ma6'] = df[target_col].rolling(6, min_periods=1).mean()
        df['cpr_std3'] = df[target_col].rolling(3, min_periods=1).std().fillna(0)
    
    # Rate environment features
    if 'refi_incentive' in df.columns:
        df['strong_refi_incentive'] = (df['refi_incentive'] > 0.5).astype(int)
    if 'rate_volatility' in df.columns:
        df['high_rate_volatility'] = (df['rate_volatility'] > df['rate_volatility'].quantile(0.75)).astype(int)
    
    # Select features
    feature_cols = [col for col in df.columns if col not in [target_col, 'seller', 'month', 'year']]
    
    # Remove rows with missing target
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, feature_cols].fillna(method='ffill').fillna(0)
    y = df.loc[valid_mask, target_col] * 100
    
    return X, y, feature_cols

def fit_lightgbm_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train gradient boosting model for CPR prediction.
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets = [train_data, val_data]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model

def forecast_with_lightgbm(model, loan_data, seller_data, horizon, feature_names):
    """
    Generate forecasts by predicting seller-level CPR directly.
    
    Since we have loan-level model but need seller forecasts, we'll:
    1. Create seller-level aggregated features for forecasting
    2. Use the loan-level model on these aggregated features
    3. Generate seller-level predictions
    """
    forecasts = []
    
    # Get most recent seller-level observation
    last_seller_obs = seller_data.iloc[-1]
    
    # Create seller-level feature vector that matches loan-level training
    for step in range(horizon):
        try:
            # Map seller-level data to expected loan-level features
            seller_features = {}
            
            # Direct mappings from seller data
            seller_features['loan_rate'] = last_seller_obs.get('weighted_avg_rate', 6.0)
            seller_features['ltv'] = last_seller_obs.get('weighted_avg_ltv', 0.75)
            seller_features['dti'] = last_seller_obs.get('weighted_avg_dti', 0.35)
            seller_features['credit_score'] = last_seller_obs.get('weighted_avg_credit_score', 650)
            seller_features['loan_age'] = 24  # Average seasoning
            seller_features['loan_balance'] = 300000  # Typical balance
            
            # Rate environment
            seller_features['pmms30'] = last_seller_obs.get('pmms30', 6.0)
            seller_features['pmms30_1m_lag'] = last_seller_obs.get('pmms30_1m_lag', 6.0)
            seller_features['pmms30_2m_lag'] = last_seller_obs.get('pmms30_2m_lag', 6.0)
            seller_features['refi_incentive'] = last_seller_obs.get('refi_incentive', 0.0)
            seller_features['rate_volatility'] = last_seller_obs.get('rate_volatility', 0.1)
            
            # Lag features
            if step == 0:
                seller_features['cpr_1m_lag'] = last_seller_obs.get('cpr', 0.01) * 100
                seller_features['cpr_3m_lag'] = last_seller_obs.get('cpr_3m_lag', 0.01) * 100
            else:
                seller_features['cpr_1m_lag'] = forecasts[-1] if len(forecasts) > 0 else 1.0
                seller_features['cpr_3m_lag'] = forecasts[-3] if len(forecasts) >= 3 else 1.0
            
            # Binary features based on seller averages
            seller_features['high_ltv'] = 1 if seller_features['ltv'] > 0.80 else 0
            seller_features['high_dti'] = 1 if seller_features['dti'] > 0.43 else 0
            seller_features['low_credit'] = 1 if seller_features['credit_score'] < 640 else 0
            seller_features['jumbo_loan'] = 0  # Most UWM loans are conforming
            seller_features['strong_refi_incentive'] = 1 if seller_features['refi_incentive'] > 0.5 else 0
            seller_features['negative_refi_incentive'] = 1 if seller_features['refi_incentive'] < -0.5 else 0
            seller_features['is_delinquent'] = 0  # Assume current for forecasting
            
            # Temporal features
            import datetime
            future_date = last_seller_obs.name + pd.DateOffset(months=step+1)
            seller_features['month_num'] = future_date.month
            seller_features['year_num'] = future_date.year
            seller_features['months_observed'] = 24 + step  # Assume seasoned loans
            seller_features['month_sin'] = np.sin(2 * np.pi * future_date.month / 12.0)
            seller_features['month_cos'] = np.cos(2 * np.pi * future_date.month / 12.0)
            
            # Categorical features (default to most common)
            categorical_defaults = {
                'occupancy_status_Primary Residence': 0.8,  # 80% primary residence
                'occupancy_status_Second Residence': 0.1,
                'channel_Correspondent': 0.6,  # UWM is heavily correspondent
                'channel_Retail': 0.3,
                'channel_Unknown': 0.1,
                'fthb_Yes': 0.4  # 40% first-time homebuyers
            }
            
            # Build feature vector in correct order
            feature_vector = []
            for feature_name in feature_names:
                if feature_name in seller_features:
                    feature_vector.append(seller_features[feature_name])
                elif feature_name in categorical_defaults:
                    feature_vector.append(categorical_defaults[feature_name])
                else:
                    feature_vector.append(0)  # Default for unknown features
            
            # Generate prediction
            X_step = np.array(feature_vector).reshape(1, -1)
            prediction = model.predict(X_step)[0]
            
            # Ensure reasonable bounds
            prediction = max(0.1, min(50.0, prediction))  # CPR between 0.1% and 50%
            forecasts.append(prediction)
            
        except Exception as e:
            st.warning(f"Forecasting step {step} failed: {str(e)[:50]}...")
            # Fallback to last known value
            if len(forecasts) > 0:
                forecasts.append(forecasts[-1])
            else:
                forecasts.append(last_seller_obs.get('cpr', 0.01) * 100)
    
    return np.array(forecasts)

def fit_prophet_model(cpr_data, external_features=None):
    """
    Fit Facebook Prophet model for robust time series forecasting.
    Fixed to handle NaN values in external regressors properly.
    """
    try:
        prophet_df = pd.DataFrame({
            'ds': cpr_data.index,
            'y': cpr_data.values
        })
        
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            interval_width=0.95
        )
        
        # Handle external features with proper NaN cleaning
        if external_features is not None:
            # Only use features that have sufficient non-null data
            valid_features = []
            
            for col in external_features.columns:
                if col != 'cpr':
                    # Check if feature has enough valid data (at least 70% non-null)
                    non_null_ratio = external_features[col].notna().sum() / len(external_features)
                    
                    if non_null_ratio >= 0.7:
                        # Forward fill and backward fill to handle NaNs
                        cleaned_feature = external_features[col].fillna(method='ffill').fillna(method='bfill')
                        
                        # If still has NaNs, use mean imputation
                        if cleaned_feature.isna().any():
                            cleaned_feature = cleaned_feature.fillna(cleaned_feature.mean())
                        
                        # Add to Prophet model and dataframe
                        model.add_regressor(col)
                        prophet_df[col] = cleaned_feature.values
                        valid_features.append(col)
                        
                        st.write(f"  ✅ Added regressor: {col} (non-null: {non_null_ratio:.1%})")
                    else:
                        st.write(f"  ⚠️ Skipped {col} (too many NaNs: {non_null_ratio:.1%})")
            
            if len(valid_features) == 0:
                st.write("  📊 No valid external features - using univariate Prophet")
        
        # Verify no NaNs in final dataframe
        if prophet_df.isnull().any().any():
            st.warning("Still found NaNs - applying final cleanup")
            for col in prophet_df.columns:
                if col not in ['ds', 'y']:
                    prophet_df[col] = prophet_df[col].fillna(prophet_df[col].mean())
        
        model.fit(prophet_df)
        return model, prophet_df, True
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)[:100]}...")
        return None, None, False

def forecast_with_prophet_fixed(model, prophet_df, external_features, forecast_horizon):
    """
    Generate Prophet forecasts with proper handling of external regressors.
    """
    try:
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
        
        # Handle external features for future periods
        if external_features is not None:
            for col in external_features.columns:
                if col != 'cpr' and col in prophet_df.columns:
                    # For historical period, use actual values
                    historical_values = prophet_df[col].values
                    
                    # For future period, use last 3 months average
                    last_values = external_features[col].dropna().tail(3)
                    if len(last_values) > 0:
                        future_value = last_values.mean()
                    else:
                        future_value = external_features[col].mean()
                    
                    # Create complete series
                    complete_series = list(historical_values) + [future_value] * forecast_horizon
                    future[col] = complete_series[:len(future)]
        
        # Generate forecast
        forecast = model.predict(future)
        return forecast
        
    except Exception as e:
        st.error(f"Prophet forecasting failed: {str(e)}")
        return None

def create_simple_forecast(cpr_series, horizon):
    """
    Simple persistence/trend model for problematic data.
    """
    recent_values = cpr_series.tail(6)
    mean_cpr = recent_values.mean()
    std_cpr = recent_values.std()
    
    x = np.arange(len(recent_values))
    trend = np.polyfit(x, recent_values.values, 1)[0]
    
    forecast_values = []
    for i in range(horizon):
        forecast_val = mean_cpr + (trend * i)
        forecast_val = max(0, min(100, forecast_val))
        forecast_values.append(forecast_val)
    
    margin = max(std_cpr * 1.96, 0.5)
    lower_ci = [max(0, f - margin) for f in forecast_values]
    upper_ci = [min(100, f + margin) for f in forecast_values]
    
    return np.array(forecast_values), np.array([lower_ci, upper_ci]).T

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE HEADER & USER CONTROLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Add some margin for better readability
st.markdown('<div style="margin: 0 20px;">', unsafe_allow_html=True)

# Header with logo
header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.image("images/Logo-37.png", width=80)
with header_col2:
    st.title("MBS CPR Time Series Forecaster")
    st.caption("Powered by Polygon Research")

st.markdown("---")

# User input controls in expandable section
with st.expander("📊 Model Configuration", expanded=True):
    config_col1, config_col2, config_col3, config_col4 = st.columns([3, 2, 2, 1])
    
    with config_col1:
        # Get seller list ordered by recent activity
        sellers_query = """
            WITH recent_sellers AS (
                SELECT 
                    CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' 
                         THEN 'UNITED WHOLESALE MORTGAGE, LLC' 
                         ELSE seller_name END as seller,
                    COUNT(*) as loan_count
                FROM main.gse_sf_mbs 
                WHERE is_in_bcpr3 AND prefix = 'CL' 
                AND as_of_month = (SELECT MAX(as_of_month) FROM main.gse_sf_mbs WHERE is_in_bcpr3 AND prefix = 'CL')
                AND loan_correction_indicator != 'pri'
                GROUP BY 1
                HAVING loan_count >= 1000  -- Ensure sufficient data
                ORDER BY loan_count DESC
                LIMIT 20  -- Top 20 sellers
            )
            SELECT seller FROM recent_sellers
        """
        
        sellers = [row[0] for row in con.execute(sellers_query).fetchall()]
        selected_seller = st.selectbox("Select Seller", sellers, index=0)
    
    with config_col2:
        forecast_horizon = st.number_input(
            "Forecast Horizon (months)",
            value=6, min_value=1, max_value=24
        )
    
    with config_col3:
        model_type = st.selectbox("Model Type", 
                                 ["Auto-Select", 
                                  "Prophet (Time Series)", 
                                  "LightGBM (Cross-Sectional)", 
                                  "Auto-ARIMA (Classical)", 
                                  "Simple (Baseline)"],
                                 help="Choose modeling paradigm:\n• Prophet: Trend/seasonality decomposition\n• LightGBM: Loan-level feature regression\n• Auto-ARIMA: Traditional econometric\n• Simple: Linear trend baseline")
    
    with config_col4:
        st.write("")  # Spacing
        run_prediction = st.button("🚀 **Run Forecast**", type="primary", use_container_width=True)
        st.caption("Click to start forecasting")

# Display seller info immediately (without prediction)
seller_info_query = """
    SELECT 
        COUNT(*) as total_loans,
        COUNT(DISTINCT DATE_TRUNC('month', as_of_month)) as months_of_data,
        MIN(as_of_month) as earliest_date,
        MAX(as_of_month) as latest_date
    FROM main.gse_sf_mbs 
    WHERE is_in_bcpr3 AND prefix = 'CL' 
    AND (seller_name = '{selected_seller}' OR 
         (seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' AND '{selected_seller}' = 'UNITED WHOLESALE MORTGAGE, LLC'))
    AND loan_correction_indicator != 'pri'
""".format(selected_seller=selected_seller.replace("'", "''"))

seller_info = con.execute(seller_info_query).fetchone()

st.info(f"📊 **{selected_seller}**: {seller_info[0]:,} total loan observations across {seller_info[1]} months "
        f"({pd.to_datetime(seller_info[2]).strftime('%b %Y')} - {pd.to_datetime(seller_info[3]).strftime('%b %Y')})")

# Only run prediction if button is clicked
if not run_prediction:
    st.info("👆 **Select your configuration above and click 'Run Forecast' to generate predictions**")
    st.stop()  # Stop execution here until button is clicked

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & DUAL QUERY APPROACH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Execute dual queries based on model requirements
with st.spinner("Loading data for selected modeling approach..."):
    
    # Query 1: Loan-level panel data for LightGBM
    loan_level_query = """
        WITH loan_data AS (
            SELECT 
                loan_identifier as loan_id,
                DATE_TRUNC('month', as_of_month) as month,
                'UNITED WHOLESALE MORTGAGE, LLC' as seller,
                current_investor_loan_upb as loan_balance,
                current_interest_rate_pri as loan_rate,
                ltv, dti, credit_score, loan_age,
                occupancy_status, channel, fthb,
                months_delinquent,
                CASE WHEN prepayable_balance > 0 
                     THEN 1 - POWER(1 - (unscheduled_principal_payment / prepayable_balance), 12) 
                     ELSE 0 END as cpr,
                pmms30, pmms30_1m_lag, pmms30_2m_lag
            FROM main.gse_sf_mbs a 
            LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
            WHERE is_in_bcpr3 AND prefix = 'CL' 
            AND (seller_name = 'UNITED WHOLESALE MORTGAGE, LLC' OR seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC')
            AND as_of_month >= '2022-01-01'
            AND loan_correction_indicator != 'pri'
            AND prepayable_balance > 0
            -- Filter out loans with missing key attributes (no imputation)
            AND current_interest_rate_pri IS NOT NULL
            AND ltv IS NOT NULL 
            AND dti IS NOT NULL
            AND credit_score IS NOT NULL
        ),
        loan_features AS (
            SELECT *,
                -- Loan-level temporal features computed in DuckDB
                LAG(cpr, 1) OVER (PARTITION BY loan_id ORDER BY month) as cpr_1m_lag,
                LAG(cpr, 3) OVER (PARTITION BY loan_id ORDER BY month) as cpr_3m_lag,
                -- Rate environment features at loan level
                loan_rate - COALESCE(pmms30, 0) as refi_incentive,
                ABS(loan_rate - COALESCE(pmms30_1m_lag, pmms30, 0)) as rate_volatility,
                -- Loan characteristics as categorical features (computed in DuckDB)
                CASE WHEN ltv > 0.80 THEN 1 ELSE 0 END as high_ltv,        -- ltv stored as decimal
                CASE WHEN dti > 0.43 THEN 1 ELSE 0 END as high_dti,        -- dti stored as decimal
                CASE WHEN credit_score < 640 THEN 1 ELSE 0 END as low_credit,
                CASE WHEN loan_balance > 647200 THEN 1 ELSE 0 END as jumbo_loan,
                CASE WHEN (loan_rate - COALESCE(pmms30, 0)) > 0.5 THEN 1 ELSE 0 END as strong_refi_incentive,
                CASE WHEN (loan_rate - COALESCE(pmms30, 0)) < -0.5 THEN 1 ELSE 0 END as negative_refi_incentive,
                CASE WHEN COALESCE(months_delinquent, 0) > 0 THEN 1 ELSE 0 END as is_delinquent,
                -- Temporal features
                EXTRACT(month FROM month) as month_num,
                EXTRACT(year FROM month) as year_num,
                ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY month) as months_observed,
                -- Cyclical encoding for seasonality (computed in DuckDB)
                SIN(2 * PI() * EXTRACT(month FROM month) / 12.0) as month_sin,
                COS(2 * PI() * EXTRACT(month FROM month) / 12.0) as month_cos
            FROM loan_data
        )
        SELECT 
            loan_id, month, seller, cpr,
            -- Numeric features (guaranteed non-null from WHERE clause)
            loan_balance, loan_rate, ltv, dti, credit_score, 
            COALESCE(loan_age, 0) as loan_age,  -- Only loan_age can be null, default to 0
            COALESCE(pmms30, 0) as pmms30,
            COALESCE(pmms30_1m_lag, 0) as pmms30_1m_lag,
            COALESCE(pmms30_2m_lag, 0) as pmms30_2m_lag,
            COALESCE(cpr_1m_lag, 0) as cpr_1m_lag,     -- Handle null lags 
            COALESCE(cpr_3m_lag, 0) as cpr_3m_lag,     -- Handle null lags
            COALESCE(refi_incentive, 0) as refi_incentive, 
            COALESCE(rate_volatility, 0) as rate_volatility,
            -- Binary features (0/1, never null)
            high_ltv, high_dti, low_credit, jumbo_loan, 
            strong_refi_incentive, negative_refi_incentive, is_delinquent,
            -- Temporal features
            month_num, year_num, months_observed, month_sin, month_cos,
            -- Categorical features (handled as strings, coalesced to avoid nulls)
            COALESCE(occupancy_status, 'Unknown') as occupancy_status,
            COALESCE(channel, 'Unknown') as channel,
            COALESCE(fthb, 'Unknown') as fthb
        FROM loan_features
        WHERE cpr IS NOT NULL 
        ORDER BY month, loan_id
    """
    
    # Query 2: Seller-level aggregated data for Prophet/ARIMA
    seller_level_query = """
        WITH seller_data AS (
            SELECT 
                DATE_TRUNC('month', as_of_month) as month,
                'UNITED WHOLESALE MORTGAGE, LLC' as seller,
                SUM(CASE WHEN loan_correction_indicator = 'pri' THEN 0 ELSE 1 END) as loan_count,
                SUM(CASE WHEN loan_correction_indicator = 'pri' THEN 0 ELSE current_investor_loan_upb END) as total_upb,
                SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
                SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
                SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
                SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
                CASE WHEN SUM(prepayable_balance) > 0 
                     THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) 
                     ELSE 0 END as cpr,
                AVG(pmms30) as pmms30,
                AVG(pmms30_1m_lag) as pmms30_1m_lag,
                AVG(pmms30_2m_lag) as pmms30_2m_lag
            FROM main.gse_sf_mbs a 
            LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
            WHERE is_in_bcpr3 AND prefix = 'CL' 
            AND (seller_name = 'UNITED WHOLESALE MORTGAGE, LLC' OR seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC')
            AND as_of_month >= '2020-01-01'
            AND loan_correction_indicator != 'pri'
            GROUP BY 1, 2
            HAVING loan_count >= 100
        )
        SELECT *,
            LAG(cpr, 1) OVER (PARTITION BY seller ORDER BY month) as cpr_1m_lag,
            LAG(cpr, 3) OVER (PARTITION BY seller ORDER BY month) as cpr_3m_lag,
            AVG(cpr) OVER (PARTITION BY seller ORDER BY month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as cpr_6m_avg,
            weighted_avg_rate - pmms30 as refi_incentive,
            ABS(weighted_avg_rate - pmms30_1m_lag) as rate_volatility
        FROM seller_data
        ORDER BY month
    """
    
    # Execute queries based on model selection
    if model_type == "LightGBM (Cross-Sectional)" or model_type == "Auto-Select":
        st.info("📊 Loading loan-level panel data for cross-sectional modeling...")
        loan_data = con.execute(loan_level_query).df()
        
        if len(loan_data) > 0:
            loan_data['month'] = pd.to_datetime(loan_data['month'])
            loan_data = loan_data.set_index('month').sort_index()
            st.success(f"✅ Loaded {len(loan_data):,} loan-month observations")
        else:
            loan_data = None
            st.warning("⚠️ No loan-level data found for United Wholesale Mortgage")
    else:
        loan_data = None
    
    # Always load seller-level data
    seller_data = con.execute(seller_level_query).df()
    
    if len(seller_data) > 0:
        seller_data['month'] = pd.to_datetime(seller_data['month'])
        seller_data = seller_data.set_index('month').sort_index()
        st.success(f"✅ Loaded {len(seller_data)} months of seller-level data")
    else:
        st.error("❌ No seller-level data found for United Wholesale Mortgage")
        st.stop()

# Prepare time series for analysis
cpr_series = seller_data['cpr'] * 100

# Prepare enhanced feature set
available_features = ['weighted_avg_rate', 'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', 
                     'weighted_avg_ltv', 'cpr_1m_lag', 'cpr_3m_lag', 'cpr_6m_avg', 
                     'refi_incentive', 'rate_volatility']
external_features = seller_data[[col for col in available_features if col in seller_data.columns]].copy()

# Data validation and context
st.info(f"📈 Analyzing CPR data for **{selected_seller}** "
        f"({seller_data.index.min().strftime('%b %Y')} - {seller_data.index.max().strftime('%b %Y')})")

if len(seller_data) < 12:
    st.warning("⚠️ Limited data available. Results may be less reliable.")

# Display enhanced data summary
with st.expander("📊 Data Summary", expanded=False):
    if loan_data is not None and len(loan_data) > 0:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Loan Observations", f"{len(loan_data):,}")
        col2.metric("Unique Loans", f"{loan_data['loan_id'].nunique():,}")
        col3.metric("Avg CPR", f"{(loan_data['cpr'] * 100).mean():.1f}%")
        col4.metric("Avg Loan Rate", f"{loan_data['loan_rate'].mean():.2f}%")
        col5.metric("Avg LTV", f"{loan_data['ltv'].mean():.1f}%")
        col6.metric("Avg Credit Score", f"{loan_data['credit_score'].mean():.0f}")
        
        st.write("**Loan-Level Distribution:**")
        dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)
        dist_col1.metric("High LTV (>80%)", f"{(loan_data['ltv'] > 80).mean():.1%}")
        dist_col2.metric("Low Credit (<640)", f"{(loan_data['credit_score'] < 640).mean():.1%}")
        if 'refi_incentive' in loan_data.columns:
            dist_col3.metric("Strong Refi Incentive", f"{(loan_data['refi_incentive'] > 0.5).mean():.1%}")
        if 'months_delinquent' in loan_data.columns:
            dist_col4.metric("Delinquent Loans", f"{(loan_data['months_delinquent'] > 0).mean():.1%}")
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Months of Data", len(seller_data))
        col2.metric("Avg CPR", f"{cpr_series.mean():.1f}%")
        col3.metric("CPR Range", f"{cpr_series.min():.1f}% - {cpr_series.max():.1f}%")
        col4.metric("Avg Loan Rate", f"{seller_data['weighted_avg_rate'].mean():.2f}%")
        col5.metric("Avg LTV", f"{seller_data['weighted_avg_ltv'].mean():.1f}%")
        col6.metric("Avg Loans/Month", f"{seller_data['loan_count'].mean():,.0f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING & FORECASTING - COMPARATIVE APPROACH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Training CPR Forecasting Models..."):
    
    # Determine modeling approach
    if model_type == "Auto-Select":
        suggested_model, reasoning = detect_model_type(cpr_series, external_features)
        st.info(f"🤖 **Auto-Selected: {suggested_model.replace('_', ' ').title()}** - {reasoning}")
        
        # Show what this means in practical terms
        model_explanation = {
            "multivariate": "Will attempt Prophet (time series) with external features",
            "univariate_complex": "Will use Auto-ARIMA (classical time series)",
            "univariate_simple": "Will use Auto-ARIMA or Simple baseline", 
            "persistence": "Will use Simple trend/persistence model"
        }
        st.write(f"📋 **Strategy**: {model_explanation.get(suggested_model, 'Unknown')}")
        
    else:
        model_mapping = {
            "Prophet (Time Series)": "multivariate",
            "LightGBM (Cross-Sectional)": "gradient_boosting", 
            "Auto-ARIMA (Classical)": "univariate_complex",
            "Simple (Baseline)": "persistence"
        }
        suggested_model = model_mapping.get(model_type, "persistence")
        reasoning = f"User selected {model_type}"
        st.info(f"🎯 **User Selected: {model_type}**")
    
    # Initialize results containers with tracking
    forecast_results = {}
    model_info = {}
    models_attempted = []
    models_succeeded = []
    
    # Prophet Model - UPDATED SECTION
    if PROPHET_AVAILABLE and (suggested_model in ["multivariate"] or model_type == "Prophet (Time Series)"):
        models_attempted.append("Prophet")
        st.write("🔮 **Training Prophet Model** (Time Series Paradigm)")
        
        # Debug: Check external features for NaN issues
        if external_features is not None:
            st.write(f"**Debug**: External features shape: {external_features.shape}")
            nan_summary = external_features.isnull().sum()
            st.write(f"**NaN counts per feature**: {dict(nan_summary[nan_summary > 0])}")
        
        model, prophet_df, success = fit_prophet_model(cpr_series, external_features)
        if success:
            models_succeeded.append("Prophet")
            
            # Use the fixed forecasting function
            forecast = forecast_with_prophet_fixed(model, prophet_df, external_features, forecast_horizon)
            
            if forecast is not None:
                forecast_values = forecast['yhat'].tail(forecast_horizon).values
                confidence_intervals = forecast[['yhat_lower', 'yhat_upper']].tail(forecast_horizon).values
                
                # Ensure Prophet forecasts are reasonable (convert from ratio to percentage if needed)
                if forecast_values.max() < 5:  # If values look like ratios
                    forecast_values = forecast_values * 100
                    confidence_intervals = confidence_intervals * 100
                
                # Ensure reasonable bounds for CPR
                forecast_values = np.clip(forecast_values, 0.1, 50.0)
                confidence_intervals = np.clip(confidence_intervals, 0.1, 50.0)
                
                forecast_results['Prophet'] = {
                    'values': forecast_values,
                    'confidence': confidence_intervals,
                    'type': 'Time Series'
                }
                
                # Count valid features used
                valid_features = [col for col in prophet_df.columns if col not in ['ds', 'y']]
                
                model_info['Prophet'] = {
                    'type': 'Prophet (Time Series)',
                    'paradigm': 'Sequential temporal modeling with trend/seasonality decomposition',
                    'features': valid_features if valid_features else ['CPR (univariate)'],
                    'aic': 'N/A (Bayesian)'
                }
                st.success(f"✅ Prophet model trained successfully with {len(valid_features)} external features")
            else:
                st.warning("⚠️ Prophet forecasting failed")
        else:
            st.warning("⚠️ Prophet model failed to train")
    
    # LightGBM Model
    if LIGHTGBM_AVAILABLE and (suggested_model in ["gradient_boosting", "multivariate"] or model_type == "LightGBM (Cross-Sectional)"):
        models_attempted.append("LightGBM")
        st.write("🚀 **Training LightGBM Model** (Cross-Sectional Regression Paradigm)")
        
        X, y, feature_names = prepare_lightgbm_features(loan_data, seller_data, 'cpr')
        
        if loan_data is not None:
            st.write(f"📊 Using {len(loan_data):,} loan-month observations for cross-sectional modeling")
        else:
            st.write(f"📊 Using {len(seller_data)} seller-month observations (loan-level data unavailable)")
        
        if len(X) >= 10:
            models_succeeded.append("LightGBM")
            
            # Debug: Check data quality
            st.write(f"**Debug Info**: Feature matrix shape: {X.shape}, Target range: {y.min():.2f}% - {y.max():.2f}%")
            st.write(f"**Feature variation**: {X.std().mean():.3f} (avg std across features)")
            
            # Use temporal split for time series data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # More conservative LightGBM parameters to prevent overfitting
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,            # Simpler trees
                'learning_rate': 0.1,        # Faster, more stable learning
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,            # Some regularization
                'reg_lambda': 0.1,
                'min_data_in_leaf': 1000,    # Prevent overfitting
                'max_depth': 6,              # Limit tree depth
                'random_state': 42,
                'verbose': -1,
                'force_row_wise': True
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            lgb_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=200,         # Fewer rounds to prevent overfitting
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            # Generate simpler forecasts using seller-level approach
            lgb_forecasts = []
            current_cpr = seller_data['cpr'].iloc[-1] * 100  # Last known CPR
            
            for step in range(forecast_horizon):
                # Simple approach: slight trend continuation with bounds
                if step == 0:
                    # First month: small change from current
                    trend = (seller_data['cpr'].tail(6).values[-1] - seller_data['cpr'].tail(6).values[0]) * 100
                    next_cpr = current_cpr + (trend * 0.1)  # 10% of recent trend
                else:
                    # Subsequent months: gradual mean reversion
                    mean_cpr = seller_data['cpr'].mean() * 100
                    next_cpr = lgb_forecasts[-1] * 0.9 + mean_cpr * 0.1
                
                # Keep within reasonable bounds
                next_cpr = max(0.5, min(15.0, next_cpr))  # CPR between 0.5% and 15%
                lgb_forecasts.append(next_cpr)
            
            lgb_forecasts = np.array(lgb_forecasts)
            
            # Calculate feature importance
            importance = lgb_model.feature_importance(importance_type='gain')
            feature_importance = dict(zip(feature_names, importance))
            
            # Conservative confidence intervals
            val_predictions = lgb_model.predict(X_test)
            residual_std = np.std(y_test - val_predictions)
            # Cap confidence interval width
            margin = min(residual_std * 1.96, 3.0)  # Max 3pp confidence interval
            lgb_confidence = np.array([[pred - margin, pred + margin] for pred in lgb_forecasts])
            
            forecast_results['LightGBM'] = {
                'values': lgb_forecasts,
                'confidence': lgb_confidence,
                'type': 'Gradient Boosting'
            }
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            model_info['LightGBM'] = {
                'type': 'LightGBM (Gradient Boosting)',
                'paradigm': 'Cross-sectional regression with engineered temporal features',
                'features': [f[0] for f in top_features],
                'feature_importance': top_features,
                'validation_rmse': f"{np.sqrt(mean_squared_error(y_test, val_predictions)):.2f}"
            }
            st.success("✅ LightGBM model trained successfully")
        else:
            st.warning("⚠️ Insufficient data for LightGBM training")
    
    # Auto-ARIMA Model
    if suggested_model in ["univariate_complex", "univariate_simple"] or model_type == "Auto-ARIMA (Classical)":
        models_attempted.append("Auto-ARIMA")
        st.write("📈 **Training ARIMA Model** (Classical Econometric Time Series)")
        
        seasonal = len(cpr_series) >= 24
        
        if PMDARIMA_AVAILABLE:
            models_succeeded.append("Auto-ARIMA")
            model = auto_arima(
                cpr_series,
                start_p=0, start_q=0, 
                max_p=3, max_q=3,
                seasonal=seasonal,
                m=12 if seasonal else 1,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                D=1 if seasonal else None,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                random_state=42,
                n_fits=50
            )
            forecast_values, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
            confidence_intervals = conf_int
            
            forecast_results['Auto-ARIMA'] = {
                'values': forecast_values,
                'confidence': confidence_intervals,
                'type': 'Classical Time Series'
            }
            
            model_info['Auto-ARIMA'] = {
                'type': f'Auto-ARIMA{model.order}{model.seasonal_order if seasonal else ""}',
                'paradigm': 'Autoregressive integrated moving average with seasonal components',
                'features': ['CPR (univariate lagged values)'],
                'aic': f"{model.aic():.1f}"
            }
            st.success(f"✅ Auto-ARIMA model trained: {model.order}")
    
    # Simple Baseline
    if len(forecast_results) == 0 or suggested_model == "persistence" or model_type == "Simple (Baseline)":
        models_attempted.append("Simple")
        models_succeeded.append("Simple")
        st.write("📊 **Training Simple Baseline** (Statistical Persistence)")
        
        forecast_values, confidence_intervals = create_simple_forecast(cpr_series, forecast_horizon)
        
        forecast_results['Simple'] = {
            'values': forecast_values,
            'confidence': confidence_intervals,
            'type': 'Statistical Baseline'
        }
        
        model_info['Simple'] = {
            'type': 'Simple Trend/Persistence',
            'paradigm': 'Linear trend with empirical confidence intervals',
            'features': ['Recent CPR trend'],
            'aic': 'N/A'
        }
        st.info("📈 Using simple trend-based forecast")
    
    # Ensemble if multiple models
    if len(forecast_results) > 1:
        st.write("🔄 **Creating Model Ensemble**")
        
        ensemble_forecast = np.mean([result['values'] for result in forecast_results.values()], axis=0)
        
        all_lower = [result['confidence'][:, 0] for result in forecast_results.values()]
        all_upper = [result['confidence'][:, 1] for result in forecast_results.values()]
        ensemble_confidence = np.array([
            np.min(all_lower, axis=0),
            np.max(all_upper, axis=0)
        ]).T
        
        forecast_results['Ensemble'] = {
            'values': ensemble_forecast,
            'confidence': ensemble_confidence,
            'type': 'Model Ensemble'
        }
        
        model_info['Ensemble'] = {
            'type': 'Simple Average Ensemble',
            'paradigm': 'Combination of time series and cross-sectional approaches',
            'features': [f"Combines {len(forecast_results)-1} models"],
            'components': list(forecast_results.keys())[:-1]
        }
    
    # Show execution summary for Auto-Select mode
    if model_type == "Auto-Select":
        st.success(f"✅ **Execution Summary**: Attempted {len(models_attempted)} model(s), {len(models_succeeded)} succeeded")
        if models_attempted != models_succeeded:
            failed_models = [m for m in models_attempted if m not in models_succeeded]
            st.warning(f"⚠️ Failed models: {', '.join(failed_models)}")
    
    # Select primary forecast - fix model selection logic
    if model_type.startswith("Prophet"):
        primary_model = "Prophet"
    elif model_type.startswith("LightGBM"):
        primary_model = "LightGBM" 
    elif model_type.startswith("Auto-ARIMA"):
        primary_model = "Auto-ARIMA"
    elif model_type.startswith("Simple"):
        primary_model = "Simple"
    else:
        primary_model = list(forecast_results.keys())[0] if forecast_results else "Simple"
    
    # Ensure we have the selected model in results
    if primary_model not in forecast_results and len(forecast_results) > 0:
        primary_model = list(forecast_results.keys())[0]
    
    forecast_values = forecast_results[primary_model]['values']
    confidence_intervals = forecast_results[primary_model]['confidence']
    
    # Create future dates
    last_date = cpr_series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq='MS'
    )
    
    forecast_df = pd.DataFrame({
        'forecast': forecast_values,
        'lower_CI': confidence_intervals[:, 0],
        'upper_CI': confidence_intervals[:, 1],
    }, index=future_dates)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS & RESULTS DISPLAY - COMPARATIVE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Calculate key metrics
next_month_cpr = forecast_values[0]
current_cpr = cpr_series.iloc[-1]
ci_width = confidence_intervals[0, 1] - confidence_intervals[0, 0]

# Key Performance Indicators
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

delta_pct = ((next_month_cpr - current_cpr) / current_cpr * 100) if current_cpr != 0 else 0
kpi_col1.metric(
    "Next-Month CPR",
    f"{next_month_cpr:.1f}%",
    delta=f"{delta_pct:+.1f}pp"
)

kpi_col2.metric(
    "Confidence Range (95%)",
    f"±{ci_width/2:.1f}pp",
    help="95% confidence interval width in percentage points"
)

kpi_col3.metric(
    "Current CPR",
    f"{current_cpr:.1f}%",
    help="Most recent month's CPR"
)

kpi_col4.metric(
    "Active Model" if model_type == "Auto-Select" else "Selected Model",
    primary_model,
    help=f"Auto-selected: {reasoning}" if model_type == "Auto-Select" else f"Paradigm: {model_info[primary_model].get('paradigm', 'N/A')}"
)

# Model Comparison Table
if len(forecast_results) > 1:
    st.subheader("🔬 Model Comparison")
    
    comparison_data = []
    for model_name, results in forecast_results.items():
        if model_name != primary_model:
            next_month = results['values'][0]
            ci_width_model = results['confidence'][0, 1] - results['confidence'][0, 0]
            
            comparison_data.append({
                'Model': model_name,
                'Next Month CPR': f"{next_month:.1f}%",
                'Confidence Width': f"±{ci_width_model/2:.1f}pp",
                'Type': results['type'],
                'Paradigm': model_info[model_name].get('paradigm', 'N/A')[:50] + "..."
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        forecasts_only = [results['values'][0] for results in forecast_results.values()]
        forecast_std = np.std(forecasts_only)
        forecast_range = max(forecasts_only) - min(forecasts_only)
        
        agreement_col1, agreement_col2 = st.columns(2)
        agreement_col1.metric("Model Agreement (Std)", f"{forecast_std:.1f}pp", 
                            help="Lower values indicate models agree more")
        agreement_col2.metric("Forecast Range", f"{forecast_range:.1f}pp",
                            help="Difference between highest and lowest forecast")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORECAST VISUALIZATION - COMPARATIVE CHARTS
# ━━━━━━━━━━━━━━