import marimo as mo

# 1. Define the application object
app = mo.App(width="full")

# 2. Define each part of our application as a cell
# ==============================================================================
# CELL 1: Imports
# ==============================================================================
@app.cell
def __creator__():
    import duckdb
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import os
    import marimo as mo
    return duckdb, mo, os, pd, np, datetime, timedelta, px, go

# ==============================================================================
# CELL 2: App Title
# ==============================================================================
@app.cell
def __(mo):
    return mo.md("# 📈 MBS CPR Time Series Forecasting Tool")

# ==============================================================================
# CELL 3: Database Connection
# ==============================================================================
@app.cell
def __(duckdb, mo, os):
    db_path = os.path.expanduser(os.getenv('MBS_DB_PATH', '~/data2/mbs/mbs.db'))
    db_path_display = mo.md(f"**Database:** `{db_path}`")
    con = duckdb.connect(db_path, read_only=True)
    return con, db_path_display

# ==============================================================================
# CELL 4: UI Control Creation
# ==============================================================================
@app.cell
def __(mo):
    scenario = mo.ui.dropdown(
        options=["Normal", "Recession", "Expansion", "Crisis"],
        value="Normal", label="Economic Scenario"
    )
    forecast_months = mo.ui.slider(
        start=3, stop=24, step=3, value=12, label="Forecast Horizon (months)"
    )
    rate_sensitivity = mo.ui.slider(
        start=-5.0, stop=0.0, step=0.5, value=-2.0,
        label="Rate Sensitivity (CPR per 100bp drop)"
    )
    return forecast_months, rate_sensitivity, scenario

# ==============================================================================
# CELL 5: UI Control Layout
# ==============================================================================
@app.cell
def __(forecast_months, mo, rate_sensitivity, scenario):
    controls_layout = mo.hstack(
        [scenario, forecast_months, rate_sensitivity],
        justify="start", gap=3
    )
    return controls_layout

# ==============================================================================
# CELL 6: Data Query Function
# ==============================================================================
@app.cell
def __(con, pd):
    query = """
    SELECT as_of_month as Date,
        CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' THEN 'UNITED WHOLESALE MORTGAGE, LLC' ELSE seller_name END as seller,
        agency, channel, fthb, occupancy_status,
        sum(case when loan_correction_indicator = 'pri' then 0 else 1 end) as loan_count,
        sum(case when loan_correction_indicator = 'pri' then 0 else current_investor_loan_upb end) as total_upb,
        SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
        SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
        SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
        SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
        CASE WHEN SUM(prepayable_balance) > 0 THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) ELSE 0 END as cpr,
        b.pmms30, b.pmms30_1m_lag, b.pmms30_2m_lag
    FROM main.gse_sf_mbs a 
    LEFT JOIN main.pmms b 
        ON a.as_of_month = b.as_of_date
    WHERE is_in_bcpr3 AND prefix = 'CL' AND seller_name IN ('UNITED SHORE FINANCIAL SERVICES, LLC', 'UNITED WHOLESALE MORTGAGE, LLC')
    AND as_of_month >= '2022-01-01'
    GROUP BY ALL ORDER BY as_of_month;
    """
    historical_data = con.sql(query).df()
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data['cpr'] = pd.to_numeric(historical_data['cpr'], errors='coerce')
    historical_data = historical_data.dropna(subset=['cpr'])
    return historical_data

# ==============================================================================
# CELL 7: Forecasting Logic
# ==============================================================================
@app.cell
def __(forecast_months, historical_data, np, pd, rate_sensitivity, scenario, timedelta):
    # This cell contains all your custom forecasting functions
    multipliers = {"Normal": 1.0, "Recession": 0.6, "Expansion": 1.3, "Crisis": 0.3}
    cpr_series = historical_data['cpr']
    periods = forecast_months.value
    
    # Simplified Ensemble Forecast (as per your code)
    if len(cpr_series) > 1:
        mean_forecast = cpr_series.mean()
        last_val = cpr_series.iloc[-1]
        base_forecast = np.linspace(last_val, mean_forecast, periods)
    else:
        base_forecast = [0] * periods

    # Apply Economic Scenario
    multiplier = multipliers[scenario.value]
    rate_change = historical_data['weighted_avg_rate'].iloc[-1] - historical_data['weighted_avg_rate'].mean() if 'weighted_avg_rate' in historical_data else 0
    
    scenario_forecast = []
    for i, base in enumerate(base_forecast):
        rate_impact = rate_change * rate_sensitivity.value / 100 * (0.95 ** i)
        adjusted_val = base * multiplier + rate_impact
        scenario_forecast.append(max(0, min(1, adjusted_val)))

    # Create forecast dataframe
    last_date = historical_data['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(periods)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'cpr_forecast': scenario_forecast})
    
    # Confidence intervals
    historical_std = cpr_series.tail(12).std()
    forecast_df['lower_bound'] = forecast_df['cpr_forecast'] - 1.96 * historical_std
    forecast_df['upper_bound'] = forecast_df['cpr_forecast'] + 1.96 * historical_std
    forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
    
    return forecast_df

# ==============================================================================
# CELL 8: Create KPIs and Chart (Single Definition Point)
# ==============================================================================
@app.cell
def __(forecast_df, go, historical_data, mo):
    # This is the ONLY cell that defines kpi_cards and fig
    
    # --- Create KPI Statistics Cards ---
    r_squared = 0.85 # Placeholder for demo
    last_historical = historical_data['cpr'].iloc[-1]
    first_forecast = forecast_df['cpr_forecast'].iloc[0]
    trend_direction = "Falling ↘️" if first_forecast < last_historical else "Rising ↗️"

    kpi_cards = mo.hstack([
        mo.stat(label="Next Month CPR", value=f"{first_forecast:.2%}"),
        mo.stat(label="95% Confidence", value=f"±{(forecast_df['upper_bound'].iloc[0] - first_forecast):.2%}"),
        mo.stat(label="Model R² (demo)", value=f"{r_squared:.3f}"),
        mo.stat(label="Trend", value=trend_direction)
    ], justify='space-around')

    # --- Create Upgraded Interactive Chart ---
    colors = {'historical': '#3b82f6', 'forecast': '#16a34a', 'confidence_fill': 'rgba(22, 163, 74, 0.1)'}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=(forecast_df['upper_bound']*100).tolist() + (forecast_df['lower_bound']*100).tolist()[::-1],
        fill='toself', fillcolor=colors['confidence_fill'],
        line={'color': 'rgba(255,255,255,0)'}, name='95% Confidence Interval', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=historical_data['Date'], y=historical_data['cpr'] * 100,
        mode='lines+markers', name='Historical CPR',
        line={'color': colors['historical'], 'width': 2.5}, marker={'size': 4},
        hovertemplate='Date: %{x|%Y-%m}<br>CPR: %{y:.2f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], y=forecast_df['cpr_forecast'] * 100,
        mode='lines+markers', name='Forecast',
        line={'color': colors['forecast'], 'width': 2.5, 'dash': 'dash'},
        marker={'size': 6, 'symbol': 'diamond'},
        hovertemplate='Date: %{x|%Y-%m}<br>Forecast: %{y:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        title={'text': 'CPR Time Series Forecast', 'x': 0.5},
        yaxis_title='CPR (%)', yaxis_tickformat='.0f',
        template='plotly_white', height=450, hovermode='x unified',
        legend={'orientation': "h", 'yanchor': "bottom", 'y': 1.02, 'xanchor': "right", 'x': 1}
    )
    return fig, kpi_cards

# ==============================================================================
# CELL 9: Create Data Table (Single Definition Point)
# ==============================================================================
@app.cell
def __(forecast_df, historical_data, mo, pd):
    hist_display = historical_data[['Date', 'cpr']].tail(12).copy()
    hist_display['Type'] = 'Historical'
    forecast_display = forecast_df[['Date', 'cpr_forecast', 'lower_bound', 'upper_bound']].head(12).copy()
    forecast_display.rename(columns={'cpr_forecast': 'cpr'}, inplace=True)
    forecast_display['Type'] = 'Forecast'
    
    combined_data = pd.concat([hist_display, forecast_display])
    combined_data['CPR'] = combined_data['cpr'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    data_table = mo.ui.table(combined_data[['Date', 'Type', 'CPR']], page_size=24)
    return data_table

# ==============================================================================
# CELL 10: Final App Layout
# ==============================================================================
@app.cell
def __(controls_layout, data_table, db_path_display, fig, kpi_cards, mo):
    # This cell ONLY arranges the final layout.
    return mo.vstack([
        db_path_display,
        controls_layout,
        mo.ui.vspace(1), # Corrected vspace
        kpi_cards,
        mo.ui.tabs({"📈 Interactive Chart": fig, "📊 Data Table": data_table})
    ], align='center')

# ==============================================================================
# Boilerplate to run the app
# ==============================================================================
if __name__ == "__main__":
    app.run()