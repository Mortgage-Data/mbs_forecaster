import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import warnings
import traceback

warnings.filterwarnings('ignore')

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

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = e

import statsmodels.api as sm

forecast_results = None
fallback_reason = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="MBS CPR Forecaster",
    page_icon="images/Logo-37.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

con = duckdb.connect('/home/gregoliven/data2/mbs/mbs.db', read_only=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODELING FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_model_type(series, external_features=None):
    n_obs = len(series)
    series_std = series.std()
    series_mean = series.mean()
    cv = series_std / series_mean if series_mean > 0 else 0
    
    autocorr_1 = series.autocorr(lag=1) if n_obs > 1 else 0
    autocorr_3 = series.autocorr(lag=3) if n_obs > 3 else 0
    has_strong_temporal = abs(autocorr_1) > 0.3 or abs(autocorr_3) > 0.2
    
    has_variation = cv > 0.1 and series_std > 0.5
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
    df = seller_data.copy()
    
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    feature_cols = [
        'weighted_avg_rate', 'weighted_avg_ltv', 'weighted_avg_dti', 'weighted_avg_credit_score',
        'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', f'{target_col}_6m_avg',
        f'{target_col}_1m_lag', f'{target_col}_3m_lag',
        'refi_incentive', 'rate_volatility',
        'month_sin', 'month_cos', 'time_index',
        'pmms30_trend'
    ]
    
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    df = df.dropna(subset=[target_col] + available_feature_cols)
    
    X = df[available_feature_cols]
    y = df[target_col] * 100
    
    return X, y, available_feature_cols

def fit_lightgbm_model(X_train, y_train, X_val, y_val):
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
        'n_estimators': 1000
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model

def forecast_with_lightgbm(model, seller_data, horizon, feature_names, target_col='cpr'):
    last_obs = seller_data.iloc[-1:].copy()
    
    required_cols = ['time_index', 'pmms30_trend']
    missing_cols = [col for col in required_cols if col not in last_obs.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in seller_data.")
    
    forecasts = []
    last_trend_values = seller_data['pmms30_trend'].tail(3).values
    trend_slope = np.mean(np.diff(last_trend_values)) if len(last_trend_values) > 1 else 0
    
    for i in range(horizon):
        future_features = last_obs.copy()
        future_features.index = last_obs.index + pd.DateOffset(months=1)
        
        future_features['month'] = future_features.index.month
        future_features['year'] = future_features.index.year
        current_time_index = future_features['time_index'].iloc[0]
        future_features['time_index'] = current_time_index + 1
        future_features['month_sin'] = np.sin(2 * np.pi * future_features['month'] / 12)
        future_features['month_cos'] = np.cos(2 * np.pi * future_features['month'] / 12)
        
        if i == 0:
            future_features[f'{target_col}_1m_lag'] = last_obs[target_col].iloc[0]
            future_features[f'{target_col}_3m_lag'] = seller_data[target_col].iloc[-3] if len(seller_data) >= 3 else last_obs[target_col].iloc[0]
        else:
            future_features[f'{target_col}_1m_lag'] = forecasts[-1] / 100.0
            if i >= 3:
                future_features[f'{target_col}_3m_lag'] = forecasts[-3] / 100.0
            else:
                future_features[f'{target_col}_3m_lag'] = seller_data[target_col].iloc[-3+i]
        
        future_features['pmms30_trend'] = last_obs['pmms30_trend'].iloc[0] + trend_slope * (i + 1)
        
        X_step = future_features[feature_names]
        
        prediction = model.predict(X_step)[0]
        prediction = max(0.1, min(50.0, prediction))
        forecasts.append(prediction)
        
        last_obs = future_features
        last_obs[target_col] = prediction / 100.0
    
    return np.array(forecasts)

def fit_prophet_model(series, external_features=None, target_col='cpr'):
    try:
        prophet_df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        model = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=5.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        if external_features is not None:
            for col in external_features.columns:
                if col != target_col:
                    model.add_regressor(col)
                    prophet_df[col] = external_features[col].values
        
        model.fit(prophet_df)
        return model, prophet_df, True
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)[:100]}...")
        return None, None, False

def create_simple_forecast(series, horizon):
    recent_values = series.tail(6)
    mean_val = recent_values.mean()
    std_val = recent_values.std()
    
    x = np.arange(len(recent_values))
    trend = np.polyfit(x, recent_values.values, 1)[0] if len(recent_values) > 1 else 0
    
    forecast_values = []
    for i in range(horizon):
        forecast_val = mean_val + (trend * (i + 1))
        forecast_val = max(0, min(100, forecast_val))
        forecast_values.append(forecast_val)
    
    margin = max(std_val * 1.96, 0.5)
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
    config_col1, config_col2, config_col3, config_col4, config_col5, config_col6 = st.columns([3, 2, 2, 2, 2, 1])
    
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
            AND loan_correction_indicator != 'pri'
            GROUP BY 1
            HAVING COUNT(*) >= 1000
            ORDER BY COUNT(*) DESC
            LIMIT 50;
        """
        try:
            sellers = [row[0] for row in con.execute(sellers_query).fetchall()]
            uwm_name = 'UNITED WHOLESALE MORTGAGE, LLC'
            if uwm_name in sellers:
                sellers.remove(uwm_name)
            sellers.insert(0, uwm_name)
            selected_seller = st.selectbox("Select Seller", sellers, index=0)
        except Exception as e:
            st.error(f"Could not load seller list from database: {e}")
            st.stop()
    
    with config_col2:
        loan_purpose_query = """
            SELECT DISTINCT loan_purpose
            FROM main.gse_sf_mbs 
            WHERE is_in_bcpr3 AND prefix = 'CL' 
            AND as_of_month >= date_trunc('month', now()) - interval '6 months'
            AND loan_purpose IS NOT NULL
            AND loan_correction_indicator != 'pri'
            ORDER BY loan_purpose;
        """
        try:
            loan_purposes = [row[0] for row in con.execute(loan_purpose_query).fetchall()]
            loan_purposes.insert(0, "All")
            selected_loan_purpose = st.selectbox("Select Loan Purpose", loan_purposes, index=0)
        except Exception as e:
            st.error(f"Could not load loan purpose list from database: {e}")
            st.stop()
    
    with config_col3:
        forecast_horizon = st.number_input("Forecast Horizon (months)", value=6, min_value=1, max_value=24)
    
    with config_col4:
        target_variable = st.selectbox("Target Variable", ["CPR", "CPR3", "BCPR3"], help="Choose the prepayment metric to predict.")
    
    with config_col5:
        model_type = st.selectbox("Model Type", 
                                 ["Auto-Select", "LightGBM (Gradient Boosting)", "Prophet (Time Series)", "Auto-ARIMA (Classical)", "Simple (Baseline)"],
                                 help="Choose modeling paradigm.")
    
    with config_col6:
        st.write("")
        run_prediction = st.button("🚀 **Run Forecast**", type="primary", use_container_width=True)

safe_seller_name = selected_seller.replace("'", "''")
where_conditions = [
    f"(seller_name = '{safe_seller_name}' OR ('{safe_seller_name}' = 'UNITED WHOLESALE MORTGAGE, LLC' AND seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC'))"
]
if selected_loan_purpose != "All":
    safe_loan_purpose = selected_loan_purpose.replace("'", "''")
    where_conditions.append(f"loan_purpose = '{safe_loan_purpose}'")
where_clause = " AND ".join(where_conditions)

seller_info_query = f"""
    SELECT 
        COUNT(*),
        COUNT(DISTINCT DATE_TRUNC('month', as_of_month)),
        MIN(as_of_month),
        MAX(as_of_month)
    FROM main.gse_sf_mbs 
    WHERE is_in_bcpr3 AND prefix = 'CL' AND {where_clause} 
    AND loan_correction_indicator != 'pri'
    AND as_of_month >= '2022-01-01';
"""
seller_info = con.execute(seller_info_query).fetchone()
st.info(f"📊 **{selected_seller} ({selected_loan_purpose})**: {seller_info[0]:,} total loan observations across {seller_info[1]} months "
        f"({pd.to_datetime(seller_info[2]).strftime('%b %Y')} - {pd.to_datetime(seller_info[3]).strftime('%b %Y')})")

if not run_prediction:
    st.info("👆 **Select your configuration and click 'Run Forecast' to begin.**")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & QUERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Loading and preparing data..."):
    
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
        AND {where_clause}
        AND as_of_month >= '2022-01-01'
        AND loan_correction_indicator != 'pri' AND prepayable_balance > 0
        AND current_interest_rate_pri IS NOT NULL AND ltv IS NOT NULL AND credit_score IS NOT NULL
    """
    
    seller_level_query = f"""
        WITH month_scaffold AS (
            SELECT DISTINCT DATE_TRUNC('month', as_of_month) as month
            FROM main.gse_sf_mbs
            WHERE as_of_month >= '2021-10-01'
        ),
        top_sellers AS (
            SELECT
                CASE
                    WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' THEN 'UNITED WHOLESALE MORTGAGE, LLC'
                    ELSE seller_name
                END as seller_name,
                COUNT(*) as loan_count
            FROM main.gse_sf_mbs
            WHERE is_in_bcpr3 AND prefix = 'CL'
            AND as_of_month >= date_trunc('month', now()) - interval '6 months'
            AND loan_correction_indicator != 'pri' -- Exclude synthetic rows from seller sizing
            GROUP BY 1
            ORDER BY loan_count DESC
            LIMIT 100
        ),
        -- CTE for portfolio characteristics (active loans only)
        portfolio_characteristics AS (
            SELECT
                DATE_TRUNC('month', as_of_month) as month,
                SUM(1) as loan_count,
                SUM(current_investor_loan_upb) as total_upb,
                SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
                SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
                SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
                SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
                SUM(prepayable_balance) as prepayable_balance, -- Get beginning balance from active loans
                AVG(pmms30) as pmms30,
                AVG(pmms30_1m_lag) as pmms30_1m_lag,
                AVG(pmms30_2m_lag) as pmms30_2m_lag
            FROM main.gse_sf_mbs a
            LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
            WHERE is_in_bcpr3 AND prefix = 'CL' AND {where_clause}
            AND as_of_month >= '2021-10-01'
            AND loan_correction_indicator != 'pri' -- CRUCIAL: Exclude synthetic rows here
            GROUP BY DATE_TRUNC('month', as_of_month)
        ),
        -- CTE for prepayment cash flows (includes payoffs)
        prepayment_flows AS (
            SELECT
                DATE_TRUNC('month', as_of_month) as month,
                SUM(unscheduled_principal_payment) as unscheduled_principal_payment
            FROM main.gse_sf_mbs
            WHERE is_in_bcpr3 AND prefix = 'CL' AND {where_clause}
            AND as_of_month >= '2021-10-01'
            -- CRUCIAL: No exclusion here, so we capture the payoff events
            GROUP BY DATE_TRUNC('month', as_of_month)
        ),
        -- Combine characteristics and flows to calculate SMM
        base_data AS (
            SELECT
                pc.*,
                pf.unscheduled_principal_payment,
                CASE
                    WHEN pc.prepayable_balance > 0
                    THEN pf.unscheduled_principal_payment / pc.prepayable_balance
                    ELSE 0
                END as smm
            FROM portfolio_characteristics pc
            JOIN prepayment_flows pf ON pc.month = pf.month
        ),
        -- All subsequent CTEs (smm_averages, etc.) can now be refactored similarly
        -- For simplicity, let's apply the same logic to the Cohort calculations
        cohort_characteristics AS (
            SELECT
                DATE_TRUNC('month', as_of_month) as month,
                SUM(prepayable_balance) as cohort_prepayable_balance
            FROM main.gse_sf_mbs a
            WHERE is_in_bcpr3 AND prefix = 'CL'
            AND a.seller_name IN (SELECT seller_name FROM top_sellers)
            AND as_of_month >= '2021-10-01'
            AND loan_correction_indicator != 'pri' -- Exclude synthetic rows from cohort balance
            GROUP BY DATE_TRUNC('month', as_of_month)
        ),
        cohort_flows AS (
            SELECT
                DATE_TRUNC('month', as_of_month) as month,
                SUM(unscheduled_principal_payment) as cohort_unscheduled_principal
            FROM main.gse_sf_mbs a
            WHERE is_in_bcpr3 AND prefix = 'CL'
            AND a.seller_name IN (SELECT seller_name FROM top_sellers)
            AND as_of_month >= '2021-10-01'
            -- No exclusion, capture cohort payoffs
            GROUP BY DATE_TRUNC('month', as_of_month)
        ),
        cohort_data AS (
            SELECT
                cc.month,
                CASE
                    WHEN cc.cohort_prepayable_balance > 0
                    THEN cf.cohort_unscheduled_principal / cc.cohort_prepayable_balance
                    ELSE 0
                END as cohort_smm
            FROM cohort_characteristics cc
            JOIN cohort_flows cf ON cc.month = cf.month
        ),
        -- The rest of the query proceeds as before, using the correctly calculated SMM values
        smm_averages AS (
            SELECT month, smm, (smm + LAG(smm, 1) OVER (ORDER BY month) + LAG(smm, 2) OVER (ORDER BY month)) / 3 as smm3
            FROM base_data
        ),
        seller_cpr3 AS (
            SELECT month, CASE WHEN smm3 > 0 THEN 1 - POWER(1 - smm3, 12) ELSE 0 END as seller_cpr3
            FROM smm_averages
        ),
        cohort_smm_averages AS (
            SELECT month, cohort_smm, (cohort_smm + LAG(cohort_smm, 1) OVER (ORDER BY month) + LAG(cohort_smm, 2) OVER (ORDER BY month)) / 3 as cohort_smm3
            FROM cohort_data
        ),
        cohort_cpr3 AS (
            SELECT month, CASE WHEN cohort_smm3 > 0 THEN 1 - POWER(1 - cohort_smm3, 12) ELSE 0 END as cohort_cpr3
            FROM cohort_smm_averages
        ),
        seller_monthly AS (
            SELECT
                ms.month,
                bd.loan_count,
                bd.total_upb,
                bd.weighted_avg_rate,
                bd.weighted_avg_ltv,
                bd.weighted_avg_dti,
                bd.weighted_avg_credit_score,
                CASE WHEN bd.smm > 0 THEN 1 - POWER(1 - bd.smm, 12) ELSE 0 END as cpr,
                bd.smm,
                bd.pmms30,
                bd.pmms30_1m_lag,
                bd.pmms30_2m_lag,
                ROW_NUMBER() OVER (ORDER BY ms.month) - 1 as time_index,
                SIN(2 * PI() * EXTRACT(MONTH FROM ms.month) / 12) as month_sin,
                COS(2 * PI() * EXTRACT(MONTH FROM ms.month) / 12) as month_cos,
                sa.smm3,
                sc.seller_cpr3 as cpr3
            FROM month_scaffold ms
            LEFT JOIN base_data bd ON ms.month = bd.month
            LEFT JOIN smm_averages sa ON ms.month = sa.month
            LEFT JOIN seller_cpr3 sc ON ms.month = sc.month
        )
        SELECT
            sm.*,
            CASE WHEN cc.cohort_cpr3 > 0 THEN (sm.cpr3 / cc.cohort_cpr3 * 100) ELSE 0 END as bcpr3,
            LAG(sm.cpr, 1) OVER (ORDER BY sm.month) as cpr_1m_lag,
            LAG(sm.cpr, 3) OVER (ORDER BY sm.month) as cpr_3m_lag,
            AVG(sm.cpr) OVER (ORDER BY sm.month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as cpr_6m_avg,
            LAG(sm.cpr3, 1) OVER (ORDER BY sm.month) as cpr3_1m_lag,
            LAG(sm.cpr3, 3) OVER (ORDER BY sm.month) as cpr3_3m_lag,
            AVG(sm.cpr3) OVER (ORDER BY sm.month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as cpr3_6m_avg,
            LAG(CASE WHEN cc.cohort_cpr3 > 0 THEN (sm.cpr3 / cc.cohort_cpr3 * 100) ELSE 0 END, 1) OVER (ORDER BY sm.month) as bcpr3_1m_lag,
            LAG(CASE WHEN cc.cohort_cpr3 > 0 THEN (sm.cpr3 / cc.cohort_cpr3 * 100) ELSE 0 END, 3) OVER (ORDER BY sm.month) as bcpr3_3m_lag,
            AVG(CASE WHEN cc.cohort_cpr3 > 0 THEN (sm.cpr3 / cc.cohort_cpr3 * 100) ELSE 0 END) OVER (ORDER BY sm.month ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as bcpr3_6m_avg,
            LAG(sm.smm, 1) OVER (ORDER BY sm.month) as smm_1m_lag,
            LAG(sm.smm, 2) OVER (ORDER BY sm.month) as smm_2m_lag,
            AVG(sm.pmms30) OVER (ORDER BY sm.month ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as pmms30_trend,
            sm.weighted_avg_rate - sm.pmms30 as refi_incentive,
            ABS(sm.pmms30 - sm.pmms30_1m_lag) as rate_volatility
        FROM seller_monthly sm
        LEFT JOIN cohort_cpr3 cc ON sm.month = cc.month
        ORDER BY sm.month
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
    try:
        seller_data = con.execute(seller_level_query).df()
    except Exception as e:
        st.error(f"❌ SQL query failed: {str(e)}")
        st.stop()
    
    if len(seller_data) > 0:
        seller_data['month'] = pd.to_datetime(seller_data['month'])
        seller_data = seller_data.set_index('month').sort_index()
        # Keep the 2021 data for calculations, but trim it before modeling/plotting
        seller_data = seller_data.loc['2022-01-01':]
        
        target_col = target_variable.lower()
        if target_col not in seller_data.columns:
            st.error(f"❌ Target column '{target_col}' not found in data for {selected_seller} ({selected_loan_purpose}). "
                     f"This may occur if no data exists for the selected filters. "
                     f"Try selecting a different loan purpose or seller.")
            st.stop()
        
        if target_col == 'bcpr3':
            seller_data['bcpr3'] = seller_data['bcpr3'].fillna(0)
            seller_data['bcpr3_1m_lag'] = seller_data['bcpr3_1m_lag'].fillna(0)
            seller_data['bcpr3_3m_lag'] = seller_data['bcpr3_3m_lag'].fillna(0)
            seller_data['bcpr3_6m_avg'] = seller_data['bcpr3_6m_avg'].fillna(0)
        
        initial_rows = len(seller_data)
        seller_data = seller_data.dropna(subset=['cpr', 'cpr3', 'pmms30', 'weighted_avg_rate'])
        final_rows = len(seller_data)
        
        if final_rows > 0:
            st.success(f"✅ Prepared {final_rows} months of clean seller-level data for modeling.")
            if initial_rows > final_rows:
                st.info(f"ℹ️ Dropped {initial_rows - final_rows} row(s) due to missing data.")
        else:
            st.error(f"❌ After cleaning, no data remained for modeling. The time series may be too short or missing key columns for {selected_seller} ({selected_loan_purpose}).")
            st.stop()
    else:
        st.error(f"❌ No seller-level data found for {selected_seller} ({selected_loan_purpose}). Cannot proceed.")
        st.stop()

# CPR/CPR3 are decimals (e.g., 0.05), but BCPR3 is already a percentage (e.g., 110)
if target_variable == 'BCPR3':
    series = seller_data[target_col]
else:
    series = seller_data[target_col] * 100

if series.isna().all():
    st.error(f"❌ No valid data for {target_variable} in the selected period for {selected_seller} ({selected_loan_purpose}). "
             f"Please try a different loan purpose or seller.")
    st.stop()

external_features = seller_data.drop(columns=['cpr', 'cpr3', 'bcpr3', 'loan_count', 'total_upb', 'smm', 'smm_1m_lag', 'smm_2m_lag', 'smm3'], errors='ignore').copy()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING & FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Training models and generating forecasts..."):
    
    if model_type == "Auto-Select":
        suggested_model, reasoning = detect_model_type(series, external_features)
        st.info(f"🤖 **Auto-Selected Model: {suggested_model.replace('_', ' ').title()}** ({reasoning})")
    else:
        model_map = {
            "LightGBM (Gradient Boosting)": "gradient_boosting",
            "Prophet (Time Series)": "multivariate",
            "Auto-ARIMA (Classical)": "univariate_complex",
            "Simple (Baseline)": "persistence"
        }
        suggested_model = model_map.get(model_type, "persistence")

    forecast_results = {}
    model_info = {}
    fallback_reason = None
    
    if LIGHTGBM_AVAILABLE and suggested_model == "gradient_boosting":
        st.write(f"🚀 **Training LightGBM Model** (Gradient Boosting) for {target_variable}")
        try:
            X, y, feature_names = prepare_seller_level_features(seller_data, target_col)
            st.info(f"ℹ️ After preparing features for LightGBM, {len(X)} data points are available for training.")

            if len(X) > 12:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.info(f"ℹ️ Training set size: {len(X_train)}, Test set size: {len(X_test)}")
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbose': -1,
                    'n_estimators': 2000,
                    'max_depth': 10
                }
                lgb_model = lgb.LGBMRegressor(**params)
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                st.info("ℹ️ LightGBM model training completed.")
                forecast_values = forecast_with_lightgbm(lgb_model, seller_data, forecast_horizon, feature_names, target_col)
                st.info(f"ℹ️ Generated {len(forecast_values)} forecast values.")
                residuals = y_test - lgb_model.predict(X_test)
                margin = np.std(residuals) * 1.96
                conf_int = np.array([[max(0, v - margin * (1 + i * 0.1)) for i, v in enumerate(forecast_values)],
                                    [min(100, v + margin * (1 + i * 0.1)) for i, v in enumerate(forecast_values)]]).T
                forecast_results['LightGBM'] = {'values': forecast_values, 'confidence': conf_int}
                model_info['LightGBM'] = {
                    'type': 'LightGBM',
                    'paradigm': 'Gradient Boosting',
                    'score': f"RMSE: {np.sqrt(mean_squared_error(y_test, lgb_model.predict(X_test))):.2f}",
                    'features': feature_names
                }
                st.success("✅ LightGBM model trained and forecast generated.")
            else:
                fallback_reason = f"Insufficient data for LightGBM: only {len(X)} data points (need >12)."
                st.warning(f"⚠️ {fallback_reason} Falling back to Simple model.")
        except Exception as e:
            fallback_reason = f"Exception in LightGBM: {str(e)}\nTraceback: {traceback.format_exc()}"
            st.warning(f"⚠️ {fallback_reason} Falling back to Simple model.")

    elif PROPHET_AVAILABLE and suggested_model == "multivariate":
        prophet_regressors = ['refi_incentive', 'rate_volatility', 'weighted_avg_credit_score']
        final_prophet_regressors = [col for col in prophet_regressors if col in external_features.columns]
        st.write(f"🔮 **Training Prophet Model** with regressors: `{', '.join(final_prophet_regressors)}` for {target_variable}")
        model, prophet_df, success = fit_prophet_model(series, external_features[final_prophet_regressors], target_col)
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

    elif PMDARIMA_AVAILABLE and "univariate" in suggested_model:
        st.write(f"📈 **Training Auto-ARIMA Model** (Classical Econometrics) for {target_variable}")
        seasonal = len(series) >= 24
        model = auto_arima(series, seasonal=seasonal, m=12 if seasonal else 1, suppress_warnings=True, error_action='ignore', stepwise=True)
        forecast_values, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_results['Auto-ARIMA'] = {'values': forecast_values, 'confidence': conf_int}
        model_info['Auto-ARIMA'] = {'type': f'ARIMA{model.order}', 'paradigm': 'Econometric Time Series', 'score': f"{model.aic():.1f} AIC", 'features': [f'Lagged {target_variable}']}
        st.success(f"✅ Auto-ARIMA model trained: {model.order}")

    if not forecast_results:
        st.warning(f"🔄 Fallback to Simple model: No forecasts generated by selected model for {target_variable}.")
        if not LIGHTGBM_AVAILABLE:
            st.error(f"❌ LightGBM unavailable: {LIGHTGBM_IMPORT_ERROR}")
        elif fallback_reason:
            st.info(f"ℹ️ Fallback reason: {fallback_reason}")
        else:
            st.info(f"ℹ️ Fallback reason: Suggested model ({suggested_model}) did not produce results.")

        st.write(f"📊 **Using Simple Baseline Model** for {target_variable}")
        forecast_values, conf_int = create_simple_forecast(series, forecast_horizon)
        forecast_results['Simple'] = {'values': forecast_values, 'confidence': conf_int}
        model_info['Simple'] = {
            'type': 'Simple Trend',
            'paradigm': 'Statistical Baseline',
            'score': 'N/A',
            'features': ['Recent Trend']
        }

    if not forecast_results:
        st.error(f"❌ All models failed to produce a forecast for {target_variable}. Cannot continue.")
        st.stop()
    
    primary_model = list(forecast_results.keys())[0]
    forecast_df = pd.DataFrame({
        'forecast': forecast_results[primary_model]['values'],
        'lower_CI': forecast_results[primary_model]['confidence'][:, 0],
        'upper_CI': forecast_results[primary_model]['confidence'][:, 1],
    }, index=pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq='MS'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS & RESULTS DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader(f"{target_variable} Forecast Results for {selected_seller} ({selected_loan_purpose})")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
current_val = series.iloc[-1]
next_month_val = forecast_df['forecast'].iloc[0]
delta = next_month_val - current_val
ci_width = forecast_df['upper_CI'].iloc[0] - forecast_df['lower_CI'].iloc[0]

kpi_col1.metric(f"Next-Month {target_variable}", f"{next_month_val:.2f}%", f"{delta:+.2f}pp vs current")
kpi_col2.metric("Confidence Range (95%)", f"±{ci_width/2:.2f}pp", help=f"The 95% confidence interval for the next month's {target_variable} forecast.")
kpi_col3.metric(f"Current {target_variable}", f"{current_val:.2f}%", help=f"The most recently observed {target_variable} value.")
kpi_col4.metric("Active Model", primary_model, help=model_info[primary_model]['paradigm'])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORECAST VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines+markers', name=f'Historical {target_variable}', line=dict(color='#667eea', width=3), hovertemplate='%{y:.2f}%'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines+markers', name=f'{primary_model} Forecast', line=dict(color='#20c997', width=3, dash='dash'), hovertemplate='%{y:.2f}%'))


fig.add_trace(go.Scatter(x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                         y=list(forecast_df['upper_CI']) + list(forecast_df['lower_CI'][::-1]),
                         fill='toself', fillcolor='rgba(32, 201, 151, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                         hoverinfo="skip", showlegend=False, name='95% CI'))
fig.update_layout(
    template="mbs_theme",
    title=dict(text=f"{forecast_horizon}-Month {target_variable} Forecast for {selected_seller} ({selected_loan_purpose})", x=0.5, font=dict(size=24)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title=f"{target_variable} (%)"
)
st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DIAGNOSTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("🔧 Model Diagnostics & Feature Analysis"):

    st.write("#### Data Preview")
    st.write("**Seller-Level Aggregated Data (used for modeling):**")
    st.dataframe(seller_data.tail()) # Changed to .tail() to see the most recent data
    if loan_data is not None:
        st.write("**Loan-Level Sample Data (for diagnostics):**")
        st.dataframe(loan_data.head())
    st.markdown("---")

    info = model_info[primary_model]
    st.subheader(f"**{primary_model}** - {info['type']}")
    st.write(f"*Paradigm*: {info['paradigm']}")
    
    diag_cols = st.columns(3)
    diag_cols[0].metric("Performance Score", info['score'])
    diag_cols[1].metric("Features Used", len(info['features']))
    diag_cols[2].metric("Data Points", len(series))
    
    if 'LightGBM' in primary_model and 'features' in info:
        st.write("**Top 5 Features by Importance:**")
        st.json({f: int(v) for f, v in zip(info['features'], lgb_model.feature_importances_)})

st.markdown('</div>', unsafe_allow_html=True)
