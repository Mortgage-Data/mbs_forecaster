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

# Modern time series libraries
try:
    from pmdarima import auto_arima
    from pmdarima.arima import ADFTest
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    st.warning("📦 Install pmdarima for better forecasting: `pip install pmdarima`")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

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
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_model_type(cpr_series, external_features=None):
    """Determine the best modeling approach based on data characteristics"""
    n_obs = len(cpr_series)
    cpr_std = cpr_series.std()
    cpr_mean = cpr_series.mean()
    cv = cpr_std / cpr_mean if cpr_mean > 0 else 0
    
    # Check for sufficient variation and length
    has_variation = cv > 0.1 and cpr_std > 0.5  # At least 0.5% CPR variation
    sufficient_length = n_obs >= 24
    has_external = external_features is not None and len(external_features.columns) > 1
    
    if has_external and sufficient_length and has_variation:
        return "multivariate"
    elif sufficient_length and has_variation:
        return "univariate_complex" 
    elif has_variation:
        return "univariate_simple"
    else:
        return "persistence"

def fit_auto_arima_model(cpr_series, seasonal=True):
    """Fit Auto-ARIMA model with intelligent parameter selection"""
    try:
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
        return model, True
    except Exception as e:
        st.warning(f"Auto-ARIMA failed: {str(e)[:100]}...")
        return None, False

def fit_prophet_model(cpr_data, external_features=None):
    """Fit Prophet model for robust forecasting"""
    try:
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': cpr_data.index,
            'y': cpr_data.values
        })
        
        # Initialize Prophet with CPR-appropriate settings
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,  # Conservative for CPR
            interval_width=0.95
        )
        
        # Add external regressors if available
        if external_features is not None:
            for col in external_features.columns:
                if col != 'cpr':
                    model.add_regressor(col)
            
            # Add external features to prophet_df
            for col in external_features.columns:
                if col != 'cpr':
                    prophet_df[col] = external_features[col].values
        
        model.fit(prophet_df)
        return model, prophet_df, True
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)[:100]}...")
        return None, None, False

def create_simple_forecast(cpr_series, horizon):
    """Simple persistence/trend model for problematic data"""
    recent_values = cpr_series.tail(6)  # Last 6 months
    mean_cpr = recent_values.mean()
    std_cpr = recent_values.std()
    
    # Simple trend calculation
    x = np.arange(len(recent_values))
    trend = np.polyfit(x, recent_values.values, 1)[0]
    
    # Generate forecast
    forecast_values = []
    for i in range(horizon):
        forecast_val = mean_cpr + (trend * i)
        # Keep CPR in reasonable bounds (0-100%)
        forecast_val = max(0, min(100, forecast_val))
        forecast_values.append(forecast_val)
    
    # Conservative confidence intervals
    margin = max(std_cpr * 1.96, 0.5)  # At least 0.5% CPR uncertainty
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

# Fetch available sellers ordered by loan count in most recent month
sellers_query = """
    SELECT 
        DATE_TRUNC('month', as_of_month) as month,
        CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' 
             THEN 'UNITED WHOLESALE MORTGAGE, LLC' 
             ELSE seller_name END as seller,
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
    AND as_of_month >= '2020-01-01'
    GROUP BY 1, 2
    HAVING loan_count >= 100
    ORDER BY 1, 2
"""

# Get data and create seller list
full_data = con.execute(sellers_query).df()

# Create seller list ordered by recent activity
recent_month = full_data['month'].max()
recent_sellers = full_data[full_data['month'] == recent_month].sort_values('loan_count', ascending=False)
sellers = recent_sellers['seller'].tolist()

# User input controls in expandable section
with st.expander("📊 Model Configuration", expanded=True):
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        selected_seller = st.selectbox("Select Seller", sellers)
    with col_b:
        forecast_horizon = st.number_input(
            "Forecast Horizon (months)",
            value=6, min_value=1, max_value=24
        )
    with col_c:
        model_type = st.selectbox("Model Type", 
                                 ["Auto-Select", "Auto-ARIMA", "Prophet", "Simple"],
                                 help="Auto-Select chooses best model based on data characteristics")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Filter data for selected seller
seller_data = full_data[full_data['seller'] == selected_seller].copy()
seller_data = seller_data.set_index('month').sort_index()

# Prepare time series
cpr_series = seller_data['cpr'] * 100  # Convert to percentage
external_features = seller_data[['weighted_avg_rate', 'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', 'weighted_avg_ltv']].copy()

# Data validation and context
st.info(f"📈 Analyzing {len(seller_data)} months of CPR data for **{selected_seller}** "
        f"({seller_data.index.min().strftime('%b %Y')} - {seller_data.index.max().strftime('%b %Y')})")

if len(seller_data) < 12:
    st.warning("⚠️ Limited data available. Results may be less reliable.")

# Display data summary
with st.expander("📊 Data Summary", expanded=False):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg CPR", f"{cpr_series.mean():.1f}%")
    col2.metric("CPR Range", f"{cpr_series.min():.1f}% - {cpr_series.max():.1f}%")
    col3.metric("Avg Loan Rate", f"{seller_data['weighted_avg_rate'].mean():.2f}%")
    col4.metric("Avg LTV", f"{seller_data['weighted_avg_ltv'].mean():.1f}%")
    col5.metric("Avg Loans/Month", f"{seller_data['loan_count'].mean():,.0f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING & FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Training CPR Forecasting Model..."):
    
    # Determine best model approach
    if model_type == "Auto-Select":
        suggested_model = detect_model_type(cpr_series, external_features)
        st.info(f"🤖 Auto-selected model: {suggested_model.replace('_', ' ').title()}")
    else:
        model_mapping = {
            "Auto-ARIMA": "univariate_complex",
            "Prophet": "multivariate", 
            "Simple": "persistence"
        }
        suggested_model = model_mapping[model_type]
    
    # Initialize variables
    forecast_values = None
    confidence_intervals = None
    model_info = {}
    
    # Try advanced models first
    if suggested_model == "multivariate" and PROPHET_AVAILABLE:
        model, prophet_df, success = fit_prophet_model(cpr_series, external_features)
        if success:
            # Create future dataframe for Prophet
            future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
            
            # Extend external features (simple forward fill for demo)
            for col in external_features.columns:
                if col != 'cpr':
                    last_values = external_features[col].tail(3).mean()
                    future[col] = [external_features[col].iloc[i] if i < len(external_features) 
                                 else last_values for i in range(len(future))]
            
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].tail(forecast_horizon).values
            confidence_intervals = forecast[['yhat_lower', 'yhat_upper']].tail(forecast_horizon).values
            
            model_info = {
                'type': 'Prophet (Multivariate)',
                'features': list(external_features.columns),
                'aic': 'N/A'
            }
            st.success("✅ Prophet multivariate model trained successfully")
    
    # Try Auto-ARIMA if Prophet failed or wasn't selected
    if forecast_values is None and PMDARIMA_AVAILABLE and suggested_model in ["univariate_complex", "univariate_simple"]:
        seasonal = len(cpr_series) >= 24
        model, success = fit_auto_arima_model(cpr_series, seasonal=seasonal)
        if success:
            forecast_values, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
            confidence_intervals = conf_int
            
            model_info = {
                'type': f'Auto-ARIMA{model.order}{model.seasonal_order if seasonal else ""}',
                'features': ['CPR (univariate)'],
                'aic': f"{model.aic():.1f}"
            }
            st.success(f"✅ Auto-ARIMA model trained: {model.order}")
    
    # Fallback to simple model
    if forecast_values is None:
        forecast_values, confidence_intervals = create_simple_forecast(cpr_series, forecast_horizon)
        model_info = {
            'type': 'Simple Trend/Persistence',
            'features': ['Recent CPR trend'],
            'aic': 'N/A'
        }
        st.info("📈 Using simple trend-based forecast")
    
    # Create future dates
    last_date = cpr_series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq='MS'
    )
    
    # Structure forecast data
    forecast_df = pd.DataFrame({
        'forecast': forecast_values,
        'lower_CI': confidence_intervals[:, 0],
        'upper_CI': confidence_intervals[:, 1],
    }, index=future_dates)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS & RESULTS DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Calculate key metrics
next_month_cpr = forecast_values[0]
current_cpr = cpr_series.iloc[-1]
ci_width = confidence_intervals[0, 1] - confidence_intervals[0, 0]

# Key Performance Indicators
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# Next month forecast with percentage change
delta_pct = ((next_month_cpr - current_cpr) / current_cpr * 100) if current_cpr != 0 else 0
kpi_col1.metric(
    "Next-Month CPR",
    f"{next_month_cpr:.1f}%",
    delta=f"{delta_pct:+.1f}pp"
)

# Confidence interval
kpi_col2.metric(
    "Confidence Range (95%)",
    f"±{ci_width/2:.1f}pp",
    help="95% confidence interval width in percentage points"
)

# Current CPR for context
kpi_col3.metric(
    "Current CPR",
    f"{current_cpr:.1f}%",
    help="Most recent month's CPR"
)

# Model type indicator
kpi_col4.metric(
    "Model Type",
    model_info['type'].split()[0],
    help=f"Full model: {model_info['type']}"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORECAST VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create interactive chart
fig = go.Figure()

# Historical CPR data
fig.add_trace(go.Scatter(
    x=cpr_series.index,
    y=cpr_series.values,
    mode='lines+markers',
    name='Historical CPR',
    line=dict(color='#667eea', width=2),
    marker=dict(size=6),
    hovertemplate='%{x}<br>CPR: %{y:.1f}%<extra></extra>'
))

# Forecast trace
fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df['forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#20c997', width=2, dash='dash'),
    marker=dict(size=6),
    hovertemplate='%{x}<br>Forecast: %{y:.1f}%<extra></extra>'
))

# Confidence interval band
fig.add_trace(go.Scatter(
    x=list(forecast_df.index) + list(forecast_df.index[::-1]),
    y=list(forecast_df['upper_CI']) + list(forecast_df['lower_CI'][::-1]),
    fill='toself',
    fillcolor='rgba(32, 201, 151, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    showlegend=False,
    name='95% Confidence'
))

# Chart layout and styling
fig.update_layout(
    template="mbs_theme",
    title=dict(
        text=f"{forecast_horizon}-Month CPR Forecast for {selected_seller}",
        x=0.5,
        font=dict(size=24),
        pad=dict(t=20, b=20)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=14)
    ),
    xaxis=dict(
        title=dict(text="Date", font=dict(size=16)),
        showgrid=True,
        gridcolor="#eeeeee",
        tickformat="%b %Y",
        tickangle=-45,
        nticks=15,
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title=dict(text="CPR (%)", font=dict(size=16)),
        showgrid=True,
        gridcolor="#eeeeee",
        tickformat=".1f",
        tickfont=dict(size=12),
        range=[0, max(max(cpr_series), max(forecast_df['upper_CI'])) * 1.1]
    ),
    autosize=True,
    margin=dict(l=80, r=80, t=120, b=80)
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DIAGNOSTICS & ADDITIONAL INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detailed model information
with st.expander("🔧 Model Diagnostics"):
    diag_col1, diag_col2, diag_col3, diag_col4, diag_col5 = st.columns(5)
    diag_col1.metric("Model", model_info['type'], help="Statistical model used for forecasting")
    diag_col2.metric("AIC", model_info['aic'], help="Akaike Information Criterion - lower values indicate better model fit")
    diag_col3.metric("Data Points", len(cpr_series))
    diag_col4.metric("Features", len(model_info['features']), help=f"Variables used: {', '.join(model_info['features'])}")
    diag_col5.metric("Forecast Horizon", f"{forecast_horizon} months")

# Library status
with st.expander("📚 Available Libraries"):
    lib_col1, lib_col2, lib_col3 = st.columns(3)
    lib_col1.metric("pmdarima", "✅ Available" if PMDARIMA_AVAILABLE else "❌ Not Available")
    lib_col2.metric("Prophet", "✅ Available" if PROPHET_AVAILABLE else "❌ Not Available") 
    lib_col3.metric("statsmodels", "✅ Available")
    
    if not PMDARIMA_AVAILABLE:
        st.code("pip install pmdarima")
    if not PROPHET_AVAILABLE:
        st.code("pip install prophet")

# Close the margin div
st.markdown('</div>', unsafe_allow_html=True)