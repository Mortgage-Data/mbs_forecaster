# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING & FORECASTING - COMPARATIVE APPROACH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.spinner("Training CPR Forecasting Models..."):
    
    # Determine modeling approach based on user selection or auto-detection
    if model_type == "Auto-Select":
        suggested_model, reasoning = detect_model_type(cpr_series, external_features)
        st.info(f"🤖 Auto-selected model: **{suggested_model.replace('_', ' ').title()}** - {reasoning}")
    else:
        model_mapping = {
            "Prophet": "multivariate",
            "LightGBM": "gradient_boosting", 
            "Auto-ARIMA": "univariate_complex",
            "Simple": "persistence"
        }
        suggested_model = model_mapping[model_type]
        reasoning = f"User selected {model_type}"
    
    # Initialize results containers for model comparison
    forecast_results = {}
    model_info = {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIME SERIES APPROACH: Prophet Model
    # ═══════════════════════════════════════════════════════════════════════
    
    if PROPHET_AVAILABLE and suggested_model in ["multivariate"] or model_type == "Prophet":
        st.write("🔮 **Training Prophet Model** (Time Series Paradigm)")
        
        model, prophet_df, success = fit_prophet_model(cpr_series, external_features)
        if success:
            # Create future dataframe for Prophet forecasting
            future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
            
            # Project external features forward (simple forward fill for demo)
            # In production, you'd want more sophisticated external forecasts
            for col in external_features.columns:
                if col != 'cpr':
                    last_values = external_features[col].tail(3).mean()
                    future[col] = [external_features[col].iloc[i] if i < len(external_features) 
                                 else last_values for i in range(len(future))]
            
            # Generate Prophet forecast
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].tail(forecast_horizon).values
            confidence_intervals = forecast[['yhat_lower', 'yhat_upper']].tail(forecast_horizon).values
            
            # Store Prophet results
            forecast_results['Prophet'] = {
                'values': forecast_values,
                'confidence': confidence_intervals,
                'type': 'Time Series'
            }
            
            model_info['Prophet'] = {
                'type': 'Prophet (Time Series)',
                'paradigm': 'Sequential temporal modeling with trend/seasonality decomposition',
                'features': list(external_features.columns),
                'aic': 'N/A (Bayesian)'
            }
            st.success("✅ Prophet model trained successfully")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MACHINE LEARNING APPROACH: LightGBM Model  
    # ═══════════════════════════════════════════════════════════════════════
    
    if LIGHTGBM_AVAILABLE and suggested_model in ["gradient_boosting", "multivariate"] or model_type == "LightGBM":
        st.write("🚀 **Training LightGBM Model** (Cross-Sectional Regression Paradigm)")
        
        # Prepare features for gradient boosting (loan-level if available)
        X, y, feature_names = prepare_lightgbm_features(loan_data, seller_data, 'cpr')
        
        # Show data summary for LightGBM
        if loan_data is not None:
            st.write(f"📊 Using {len(loan_data):,} loan-month observations for cross-sectional modeling")
        else:
            st.write(f"📊 Using {len(seller_data)} seller-month observations (loan-level data unavailable)")
        
        if len(X) >= 10:  # Minimum data requirement
            # Time series split (respecting temporal order)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train LightGBM model
            lgb_model = fit_lightgbm_model(X_train, y_train, X_test, y_test)
            
            # Generate forecasts using recursive prediction
            last_obs = seller_data.iloc[-1]
            external_forecast = None  # Could project external features here
            
            lgb_forecasts = forecast_with_lightgbm(
                lgb_model, last_obs, external_forecast, forecast_horizon, feature_names
            )
            
            # Calculate feature importance for interpretability
            importance = lgb_model.feature_importance(importance_type='gain')
            feature_importance = dict(zip(feature_names, importance))
            
            # Estimate confidence intervals (gradient boosting doesn't provide native uncertainty)
            # Use residual standard deviation from validation set
            if len(X_test) > 0:
                val_predictions = lgb_model.predict(X_test)
                residual_std = np.std(y_test - val_predictions)
                margin = residual_std * 1.96  # 95% confidence interval
                
                lgb_confidence = np.array([[pred - margin, pred + margin] for pred in lgb_forecasts])
            else:
                # Fallback confidence interval
                margin = np.std(y.tail(12)) * 1.96 if len(y) >= 12 else 2.0
                lgb_confidence = np.array([[pred - margin, pred + margin] for pred in lgb_forecasts])
            
            # Store LightGBM results
            forecast_results['LightGBM'] = {
                'values': lgb_forecasts,
                'confidence': lgb_confidence,
                'type': 'Gradient Boosting'
            }
            
            # Get top features for reporting
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            model_info['LightGBM'] = {
                'type': 'LightGBM (Gradient Boosting)',
                'paradigm': 'Cross-sectional regression with engineered temporal features',
                'features': [f[0] for f in top_features],
                'feature_importance': top_features,
                'validation_rmse': f"{np.sqrt(mean_squared_error(y_test, val_predictions)):.2f}" if len(X_test) > 0 else "N/A"
            }
            st.success("✅ LightGBM model trained successfully")
        else:
            st.warning("⚠️ Insufficient data for LightGBM training")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLASSICAL TIME SERIES: Auto-ARIMA Approach
    # ═══════════════════════════════════════════════════════════════════════
    
    if suggested_model in ["univariate_complex", "univariate_simple"] or model_type == "Auto-ARIMA":
        st.write("📈 **Training ARIMA Model** (Classical Econometric Time Series)")
        
        seasonal = len(cpr_series) >= 24
        
        if PMDARIMA_AVAILABLE:
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
            
        else:
            # Fallback to statsmodels ARIMA
            model, params, success = fit_statsmodels_arima(cpr_series, seasonal=seasonal)
            if success:
                forecast_res = model.get_forecast(steps=forecast_horizon)
                forecast_values = forecast_res.predicted_mean.values
                confidence_intervals = forecast_res.conf_int(alpha=0.05).values
                
                forecast_results['SARIMAX'] = {
                    'values': forecast_values,
                    'confidence': confidence_intervals,
                    'type': 'Classical Time Series'
                }
                
                model_info['SARIMAX'] = {
                    'type': f'SARIMAX{params[:3]}{params[3:] if len(params) > 3 else ""}',
                    'paradigm': 'Seasonal autoregressive integrated moving average',
                    'features': ['CPR (univariate)'],
                    'aic': f"{model.aic:.1f}"
                }
                st.success(f"✅ SARIMAX model trained: {params}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FALLBACK: Simple Statistical Methods
    # ═══════════════════════════════════════════════════════════════════════
    
    # Always include simple baseline for comparison
    if len(forecast_results) == 0 or suggested_model == "persistence":
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
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENSEMBLE COMBINATION (if multiple models trained)
    # ═══════════════════════════════════════════════════════════════════════
    
    if len(forecast_results) > 1:
        st.write("🔄 **Creating Model Ensemble**")
        
        # Simple average ensemble (could be weighted based on validation performance)
        ensemble_forecast = np.mean([result['values'] for result in forecast_results.values()], axis=0)
        
        # Conservative confidence intervals (widest of all models)
        all_lower = [result['confidence'][:, 0] for result in forecast_results.values()]
        all_upper = [result['confidence'][:, 1] for result in forecast_results.values()]
        ensemble_confidence = np.array([
            np.min(all_lower, axis=0),  # Most conservative lower bound
            np.max(all_upper, axis=0)   # Most conservative upper bound
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
            'components': list(forecast_results.keys())[:-1]  # Exclude ensemble itself
        }
    
    # Select primary forecast for display (user choice or best available)
    primary_model = model_type if model_type in forecast_results else list(forecast_results.keys())[0]
    forecast_values = forecast_results[primary_model]['values']
    confidence_intervals = forecast_results[primary_model]['confidence']
    
    # Create future dates for all forecasts
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
    }, index=future_dates)# app.py
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
    # Don't show warning immediately - will show in UI if needed
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
    the time series as suitable for different modeling paradigms:
    - Temporal: Strong sequential patterns (ARIMA/Prophet)  
    - Cross-sectional: Rich features, weak temporal patterns (LightGBM)
    - Hybrid: Both temporal and cross-sectional signals present
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
    has_variation = cv > 0.1 and cpr_std > 0.5  # At least 0.5% CPR variation
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
    Feature engineering for gradient boosting using loan-level panel data.
    
    This is the key advantage of LightGBM - we can use individual loan 
    characteristics while still predicting seller-level CPR through aggregation.
    
    Args:
        loan_data: Loan-level panel data (loan_id × month observations)
        seller_data: Seller-level aggregated data for comparison
        target_col: Target variable name
    
    Returns:
        - X: Feature matrix with loan and temporal features
        - y: Target variable (loan-level CPR)
        - feature_names: List of feature names
    """
    if loan_data is None:
        # Fallback to seller-level features if loan data unavailable
        return prepare_seller_level_features(seller_data, target_col)
    
    df = loan_data.copy()
    
    # Loan-level features (cross-sectional variation)
    df['loan_age_months'] = df.get('loan_age', 0)
    df['high_ltv'] = (df['ltv'] > 80).astype(int)
    df['high_dti'] = (df['dti'] > 43).astype(int)  
    df['low_credit'] = (df['credit_score'] < 640).astype(int)
    df['jumbo_loan'] = (df['loan_balance'] > 647200).astype(int)  # 2022 conforming limit
    
    # Temporal features - convert time patterns to explicit features
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['time_index'] = df.groupby('loan_id').cumcount()  # Loan seasoning
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Rate environment features (key economic drivers)
    if 'refi_incentive' in df.columns:
        df['strong_refi_incentive'] = (df['refi_incentive'] > 0.5).astype(int)
        df['negative_refi_incentive'] = (df['refi_incentive'] < -0.5).astype(int)
    
    # Loan-level rolling statistics (within each loan)
    df = df.sort_values(['loan_id', df.index])
    if len(df) >= 6:
        df['cpr_ma3'] = df.groupby('loan_id')[target_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['loan_rate_trend'] = df.groupby('loan_id')['loan_rate'].transform(lambda x: x.diff().rolling(3).mean().fillna(0))
    
    # Select features for model
    feature_cols = [col for col in df.columns if col not in [target_col, 'loan_id', 'seller', 'month', 'year']]
    
    # Remove rows with missing target
    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, feature_cols].fillna(method='ffill').fillna(0)
    y = df.loc[valid_mask, target_col] * 100  # Convert to percentage
    
    return X, y, feature_cols

def prepare_seller_level_features(seller_data, target_col='cpr'):
    """
    Fallback feature engineering using seller-level aggregated data.
    
    When loan-level data is unavailable, we can still apply LightGBM
    to seller-level time series with engineered temporal features.
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
    
    Uses LightGBM optimized for time series regression with:
    - L2 regularization for generalization
    - Learning rate scheduling 
    - Early stopping to prevent overfitting
    
    This follows the cross-sectional regression paradigm where each
    observation is treated as independent given the feature set.
    """
    # LightGBM parameters optimized for financial time series
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,           # Moderate complexity
        'learning_rate': 0.05,      # Conservative learning
        'feature_fraction': 0.8,    # Feature sampling for robustness
        'bagging_fraction': 0.8,    # Row sampling for robustness  
        'bagging_freq': 5,
        'reg_alpha': 0.1,          # L1 regularization
        'reg_lambda': 0.1,         # L2 regularization
        'random_state': 42,
        'verbose': -1
    }
    
    # Prepare datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets = [train_data, val_data]
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model

def forecast_with_lightgbm(model, last_observation, external_forecast, horizon, feature_names):
    """
    Generate multi-step forecasts using trained LightGBM model.
    
    Uses recursive forecasting where each prediction becomes input
    for the next step. External features (rates, economic variables)
    are projected forward using simple assumptions.
    
    This is the key challenge of applying regression models to forecasting:
    we must generate our own lagged features for future periods.
    """
    forecasts = []
    current_obs = last_observation.copy()
    
    for step in range(horizon):
        # Create feature vector for this step
        X_step = current_obs[feature_names].values.reshape(1, -1)
        
        # Generate prediction
        pred = model.predict(X_step)[0]
        forecasts.append(pred)
        
        # Update observation for next step (recursive forecasting)
        # This is where regression-based forecasting gets tricky
        if 'cpr_1m_lag' in current_obs.index:
            current_obs['cpr_1m_lag'] = pred / 100  # Convert back to ratio
        if 'cpr_ma3' in current_obs.index:
            # Simple update - in practice might use more sophisticated lag updating
            current_obs['cpr_ma3'] = pred / 100
        
        # Update time-based features
        if 'time_index' in current_obs.index:
            current_obs['time_index'] += 1
        
        # Use external forecasts if available
        if external_forecast is not None and step < len(external_forecast):
            for col in external_forecast.columns:
                if col in current_obs.index:
                    current_obs[col] = external_forecast[col].iloc[step]
    
    return np.array(forecasts)

def fit_statsmodels_arima(cpr_series, seasonal=True):
    """
    Fallback ARIMA implementation using statsmodels.
    
    Performs grid search over ARIMA parameters to find optimal model.
    This is the classical econometric time series approach that models
    CPR as purely autoregressive with moving average components.
    """
    try:
        # Simple parameter search for ARIMA
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        # Test different parameter combinations
        param_combinations = [
            (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
            (1, 0, 1), (2, 0, 1), (1, 0, 2)
        ]
        
        seasonal_combinations = [(1, 1, 1, 12), (0, 1, 1, 12)] if seasonal else [(0, 0, 0, 0)]
        
        for (p, d, q) in param_combinations:
            for (P, D, Q, s) in seasonal_combinations:
                try:
                    if seasonal and len(cpr_series) >= 24:
                        model = sm.tsa.statespace.SARIMAX(
                            cpr_series,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    else:
                        model = sm.tsa.statespace.SARIMAX(
                            cpr_series,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    
                    fitted = model.fit(disp=False, maxiter=100)
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_params = (p, d, q, P, D, Q) if seasonal else (p, d, q)
                        
                except:
                    continue
        
        if best_model is not None:
            return best_model, best_params, True
        else:
            return None, None, False
            
    except Exception as e:
        return None, None, False

def fit_prophet_model(cpr_data, external_features=None):
    """
    Fit Facebook Prophet model for robust time series forecasting.
    
    Prophet uses a decomposable time series model:
    y(t) = g(t) + s(t) + h(t) + ε(t)
    
    Where:
    - g(t): Piecewise linear or logistic growth trend
    - s(t): Seasonal components (yearly, weekly, etc.)  
    - h(t): Holiday effects (if specified)
    - ε(t): Error term
    
    This approach explicitly models temporal structure while allowing
    external regressors to influence the baseline prediction.
    """
    try:
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': cpr_data.index,
            'y': cpr_data.values
        })
        
        # Initialize Prophet with CPR-appropriate settings
        model = Prophet(
            seasonality_mode='multiplicative',  # CPR seasonality scales with level
            yearly_seasonality=True,           # Seasonal refinancing patterns
            weekly_seasonality=False,          # Monthly data, no weekly patterns
            daily_seasonality=False,           # Monthly data, no daily patterns
            changepoint_prior_scale=0.1,       # Conservative trend changes
            interval_width=0.95               # 95% confidence intervals
        )
        
        # Add external regressors (key economic drivers)
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
    """
    Simple persistence/trend model for problematic data.
    
    Falls back to basic statistical methods when sophisticated models fail.
    Uses recent average with linear trend projection and empirical confidence intervals.
    """
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
                                 ["Auto-Select", "Prophet", "LightGBM", "Auto-ARIMA", "Simple"],
                                 help="Choose modeling paradigm: Prophet (time series), LightGBM (regression), or Auto-Select")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Filter data for selected seller
if 'seller_data' not in locals():
    seller_data = full_data[full_data['seller'] == selected_seller].copy()
    seller_data = seller_data.set_index('month').sort_index()

# Prepare enhanced feature set
available_features = ['weighted_avg_rate', 'pmms30', 'pmms30_1m_lag', 'pmms30_2m_lag', 
                     'weighted_avg_ltv', 'cpr_1m_lag', 'cpr_3m_lag', 'cpr_6m_avg', 
                     'refi_incentive', 'rate_volatility']
external_features = seller_data[[col for col in available_features if col in seller_data.columns]].copy()

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
        suggested_model, reasoning = detect_model_type(cpr_series, external_features)
        st.info(f"🤖 Auto-selected model: **{suggested_model.replace('_', ' ').title()}** - {reasoning}")
    else:
        model_mapping = {
            "Auto-ARIMA": "univariate_complex",
            "Prophet": "multivariate", 
            "Simple": "persistence"
        }
        suggested_model = model_mapping[model_type]
        reasoning = f"User selected {model_type}"
    
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
    if forecast_values is None and suggested_model in ["univariate_complex", "univariate_simple"]:
        seasonal = len(cpr_series) >= 24
        
        if PMDARIMA_AVAILABLE:
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
            
            model_info = {
                'type': f'Auto-ARIMA{model.order}{model.seasonal_order if seasonal else ""}',
                'features': ['CPR (univariate)'],
                'aic': f"{model.aic():.1f}"
            }
            st.success(f"✅ Auto-ARIMA model trained: {model.order}")
            
        else:
            # Fallback to statsmodels ARIMA with parameter search
            model, params, success = fit_statsmodels_arima(cpr_series, seasonal=seasonal)
            if success:
                forecast_results = model.get_forecast(steps=forecast_horizon)
                forecast_values = forecast_results.predicted_mean.values
                confidence_intervals = forecast_results.conf_int(alpha=0.05).values
                
                model_info = {
                    'type': f'SARIMAX{params[:3]}{params[3:] if len(params) > 3 else ""}',
                    'features': ['CPR (univariate)'],
                    'aic': f"{model.aic:.1f}"
                }
                st.success(f"✅ SARIMAX model trained: {params}")
            else:
                st.warning("⚠️ ARIMA model fitting failed")
    
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
# METRICS & RESULTS DISPLAY - COMPARATIVE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Calculate key metrics for primary model
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
    "Primary Model",
    primary_model,
    help=f"Paradigm: {model_info[primary_model].get('paradigm', 'N/A')}"
)

# Model Comparison Table (if multiple models trained)
if len(forecast_results) > 1:
    st.subheader("🔬 Model Comparison")
    
    comparison_data = []
    for model_name, results in forecast_results.items():
        if model_name != primary_model:  # Don't duplicate primary
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
        
        # Forecast agreement analysis
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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create interactive comparative chart
fig = go.Figure()

# Historical CPR data
fig.add_trace(go.Scatter(
    x=cpr_series.index,
    y=cpr_series.values,
    mode='lines+markers',
    name='Historical CPR',
    line=dict(color='#667eea', width=3),
    marker=dict(size=6),
    hovertemplate='%{x}<br>CPR: %{y:.1f}%<extra></extra>'
))

# Add all model forecasts for comparison
colors = ['#20c997', '#fd7e14', '#e83e8c', '#6f42c1', '#17a2b8']
line_styles = ['dash', 'dot', 'dashdot', 'longdash', 'solid']

for i, (model_name, results) in enumerate(forecast_results.items()):
    color = colors[i % len(colors)]
    line_style = line_styles[i % len(line_styles)]
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=results['values'],
        mode='lines+markers',
        name=f'{model_name} Forecast',
        line=dict(color=color, width=2, dash=line_style),
        marker=dict(size=5),
        hovertemplate=f'%{{x}}<br>{model_name}: %{{y:.1f}}%<extra></extra>',
        visible='legendonly' if model_name != primary_model else True
    ))
    
    # Confidence interval (only for primary model to avoid clutter)
    if model_name == primary_model:
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(results['confidence'][:, 1]) + list(results['confidence'][:, 0][::-1]),
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            name=f'{model_name} 95% CI'
        ))

# Enhanced chart layout
fig.update_layout(
    template="mbs_theme",
    title=dict(
        text=f"{forecast_horizon}-Month CPR Forecast Comparison for {selected_seller}",
        x=0.5,
        font=dict(size=24),
        pad=dict(t=20, b=20)
    ),
    legend=dict(
        orientation="v",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
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
        range=[0, max(max(cpr_series), max([max(r['values']) for r in forecast_results.values()])) * 1.1]
    ),
    autosize=True,
    margin=dict(l=80, r=150, t=120, b=80),
    hovermode='x unified'
)

# Display the comparative chart
st.plotly_chart(fig, use_container_width=True)

# Add explanatory text about modeling approaches
st.markdown("""
**📊 Understanding the Modeling Approaches:**

- **Prophet (Time Series)**: Captures seasonal patterns and trends automatically. Best for data with clear temporal structure.
- **LightGBM (Gradient Boosting)**: Uses feature engineering to capture complex relationships. Best for rich feature sets.
- **Auto-ARIMA (Classical)**: Traditional econometric approach focusing on autocorrelation patterns.
- **Ensemble**: Combines multiple approaches to reduce model risk and improve robustness.
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DIAGNOSTICS & DETAILED ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detailed model information
with st.expander("🔧 Model Diagnostics & Feature Analysis"):
    if len(model_info) > 1:
        # Multi-model diagnostics
        for model_name, info in model_info.items():
            st.subheader(f"**{model_name}** - {info['type']}")
            st.write(f"*Paradigm*: {info['paradigm']}")
            
            diag_cols = st.columns(4)
            diag_cols[0].metric("Features Used", len(info['features']))
            diag_cols[1].metric("AIC/Score", info.get('aic', info.get('validation_rmse', 'N/A')))
            
            if 'feature_importance' in info:
                diag_cols[2].metric("Top Feature", info['feature_importance'][0][0])
                diag_cols[3].metric("Importance", f"{info['feature_importance'][0][1]:.0f}")
                
                # Feature importance chart for LightGBM
                if model_name == 'LightGBM':
                    importance_df = pd.DataFrame(info['feature_importance'], columns=['Feature', 'Importance'])
                    st.bar_chart(importance_df.set_index('Feature')['Importance'])
            
            st.markdown("---")
    else:
        # Single model diagnostics  
        info = list(model_info.values())[0]
        st.write(f"**Model**: {info['type']}")
        st.write(f"**Paradigm**: {info['paradigm']}")
        
        diag_col1, diag_col2, diag_col3, diag_col4, diag_col5 = st.columns(5)
        diag_col1.metric("Features", len(info['features']))
        diag_col2.metric("AIC/Score", info.get('aic', 'N/A'), 
                        help="Akaike Information Criterion - lower values indicate better model fit")
        diag_col3.metric("Data Points", len(cpr_series))
        diag_col4.metric("Forecast Horizon", f"{forecast_horizon} months")
        diag_col5.metric("Data Range", f"{seller_data.index.min().strftime('%b %Y')} - {seller_data.index.max().strftime('%b %Y')}")

# Library status and installation guidance
with st.expander("📚 Available Libraries & Installation"):
    lib_col1, lib_col2, lib_col3, lib_col4 = st.columns(4)
    
    if PMDARIMA_AVAILABLE:
        lib_col1.metric("pmdarima", "✅ Available")
    else:
        lib_col1.metric("pmdarima", "❌ Not Available")
        if 'PMDARIMA_ERROR' in locals() and 'numpy.dtype size changed' in PMDARIMA_ERROR:
            st.error("🔧 **NumPy compatibility issue detected!**")
            st.code("""# Fix with these commands:
pip uninstall pmdarima -y
pip cache purge  
pip install pmdarima --no-cache-dir --force-reinstall""")
        else:
            st.code("pip install pmdarima")
    
    lib_col2.metric("Prophet", "✅ Available" if PROPHET_AVAILABLE else "❌ Not Available")
    lib_col3.metric("LightGBM", "✅ Available" if LIGHTGBM_AVAILABLE else "❌ Not Available") 
    lib_col4.metric("statsmodels", "✅ Available")
    
    if not PROPHET_AVAILABLE:
        st.code("pip install prophet")
    if not LIGHTGBM_AVAILABLE:
        st.code("pip install lightgbm scikit-learn")

# Close the margin div
st.markdown('</div>', unsafe_allow_html=True)