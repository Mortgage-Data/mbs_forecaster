import marimo

__generated_with = "0.10.12"
app = marimo.App()


@app.cell
def __():
    import duckdb
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import os
    import marimo as mo
    return duckdb, mo, os, pd, np, datetime, timedelta, px, go


@app.cell
def __(mo):
    mo.md("# 📈 MBS CPR Time Series Forecasting Tool")
    return


@app.cell
def __(duckdb, mo, os):
    db_path = os.path.expanduser(
        os.getenv('MBS_DB_PATH', '~/data/mbs.db')
    )
    db_path_display = mo.md(f"**Database:** `{db_path}`")
    con = duckdb.connect(db_path, read_only=True)
    return con, db_path, db_path_display


@app.cell
def __(db_path_display):
    db_path_display
    return


@app.cell
def __(mo):
    # Create UI controls
    scenario = mo.ui.dropdown(
        options=["Normal", "Recession", "Expansion", "Crisis"],
        value="Normal",
        label="Economic Scenario"
    )
    
    forecast_months = mo.ui.slider(
        start=3,
        stop=24,
        step=3,
        value=12,
        label="Forecast Horizon (months)"
    )
    
    rate_sensitivity = mo.ui.slider(
        start=-5.0,
        stop=0.0,
        step=0.5,
        value=-2.0,
        label="Rate Sensitivity (CPR change per 100bp rate drop)"
    )
    
    # Display controls in a nice layout
    controls = mo.hstack([
        mo.vstack([scenario, scenario.value]),
        mo.vstack([forecast_months, f"{forecast_months.value} months"]),
        mo.vstack([rate_sensitivity, f"{rate_sensitivity.value}"])
    ], justify="start", gap=3)
    
    controls
    return scenario, forecast_months, rate_sensitivity, controls


@app.cell
def __(con, scenario):
    # Query data with scenario adjustments
    multipliers = {
        "Normal": 1.0, "Recession": 0.6,
        "Expansion": 1.3, "Crisis": 0.3,
    }
    multiplier = multipliers[scenario.value]
    
    query = """
    SELECT
        as_of_month as Date,
        CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' 
             THEN 'UNITED WHOLESALE MORTGAGE, LLC' 
             ELSE seller_name END as seller,
        agency, channel, fthb, occupancy_status,
        sum(case when loan_correction_indicator = 'pri' then 0 else 1 end) as loan_count,
        sum(case when loan_correction_indicator = 'pri' then 0 else current_investor_loan_upb end) as total_upb,
        SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
        SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
        SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
        SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
        CASE WHEN SUM(prepayable_balance) > 0 
             THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) 
             ELSE 0 END as cpr,
        b.pmms30, b.pmms30_1m_lag, b.pmms30_2m_lag
    FROM main.gse_sf_mbs a
    LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
    WHERE is_in_bcpr3 AND prefix = 'CL' 
    AND seller_name IN ('UNITED SHORE FINANCIAL SERVICES, LLC', 'UNITED WHOLESALE MORTGAGE, LLC')
    AND as_of_month >= '2022-01-01'
    GROUP BY ALL ORDER BY as_of_month;
    """
    
    historical_data = con.sql(query).df()
    
    # Ensure Date is datetime and CPR is numeric
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data['cpr'] = pd.to_numeric(historical_data['cpr'], errors='coerce')
    historical_data = historical_data.dropna(subset=['cpr'])
    
    return historical_data, multiplier, query


@app.cell
def __(historical_data, forecast_months, rate_sensitivity, scenario, multipliers, pd, np, timedelta):
    # Forecasting functions
    def calculate_arima_forecast(data, periods):
        """Simplified AR(1) model"""
        if len(data) < 2:
            return [data.iloc[-1]] * periods
        
        # Calculate lag-1 autocorrelation
        mean_val = data.mean()
        if data.std() < 0.001:  # Very low variance
            return [mean_val * 1.02] * periods  # Slight upward trend
        
        numerator = sum((data.iloc[i] - mean_val) * (data.iloc[i-1] - mean_val) 
                       for i in range(1, len(data)))
        denominator = sum((data.iloc[i] - mean_val) ** 2 
                         for i in range(len(data)))
        
        phi = numerator / denominator if denominator != 0 else 0.5
        phi = max(min(phi, 0.95), -0.95)  # Constrain phi
        
        forecast = []
        last_val = data.iloc[-1]
        
        for _ in range(periods):
            next_val = mean_val + phi * (last_val - mean_val)
            forecast.append(max(0, next_val))  # Ensure non-negative
            last_val = next_val
        
        return forecast

    def calculate_exponential_smoothing(data, periods):
        """Exponential smoothing with adaptive alpha"""
        if len(data) < 2:
            return [data.iloc[-1]] * periods
        
        # Adaptive alpha based on recent volatility
        recent_std = data.tail(6).std()
        overall_std = data.std()
        alpha = 0.3 if recent_std > overall_std * 1.5 else 0.6
        
        # Initialize
        forecast = []
        last_smoothed = data.iloc[0]
        
        # Smooth historical data
        for val in data.iloc[1:]:
            last_smoothed = alpha * val + (1 - alpha) * last_smoothed
        
        # Add trend component for flat series
        if data.std() < 0.001:
            trend = (data.iloc[-1] - data.iloc[-6]) / 6 if len(data) >= 6 else 0
        else:
            trend = 0
        
        # Forecast
        for i in range(periods):
            next_val = last_smoothed + trend * (i + 1)
            forecast.append(max(0, next_val))
        
        return forecast

    def calculate_linear_trend(data, periods):
        """Linear regression forecast"""
        if len(data) < 2:
            return [data.iloc[-1]] * periods
        
        x = np.arange(len(data))
        y = data.values
        
        # Calculate linear regression
        n = len(x)
        x_mean = x.mean()
        y_mean = y.mean()
        
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # Forecast
        forecast = []
        for i in range(periods):
            next_val = intercept + slope * (n + i)
            forecast.append(max(0, next_val))  # Ensure non-negative
        
        return forecast

    def calculate_ensemble_forecast(data, periods):
        """Average of multiple methods"""
        arima = calculate_arima_forecast(data, periods)
        exp_smooth = calculate_exponential_smoothing(data, periods)
        linear = calculate_linear_trend(data, periods)
        
        ensemble = [(a + e + l) / 3 for a, e, l in zip(arima, exp_smooth, linear)]
        return ensemble

    def apply_economic_scenario(base_forecast, scenario_name, rate_change, periods):
        """Apply economic scenario adjustments"""
        multiplier = multipliers[scenario_name]
        adjusted = []
        
        for i, base in enumerate(base_forecast):
            # Rate sensitivity impact with decay
            rate_impact = rate_change * rate_sensitivity.value / 100 * (0.95 ** i)
            
            # Seasonal adjustment
            month = (historical_data['Date'].iloc[-1].month + i) % 12 + 1
            if month in [3, 4, 5, 6, 7, 8]:  # Spring/Summer
                seasonal = 1.2
            elif month in [1, 2, 11, 12]:  # Winter
                seasonal = 0.8
            else:  # Fall
                seasonal = 1.0
            
            # Combined adjustment
            adjusted_val = base * multiplier * seasonal + rate_impact
            adjusted.append(max(0, min(1, adjusted_val)))  # Bound between 0 and 1
        
        return adjusted

    # Generate forecasts
    cpr_series = historical_data['cpr']
    periods = forecast_months.value
    
    # Base forecast
    base_forecast = calculate_ensemble_forecast(cpr_series, periods)
    
    # Calculate rate change (simplified - using last available rate vs historical average)
    if 'weighted_avg_rate' in historical_data.columns:
        current_rate = historical_data['weighted_avg_rate'].iloc[-1]
        hist_avg_rate = historical_data['weighted_avg_rate'].mean()
        rate_change = current_rate - hist_avg_rate
    else:
        rate_change = 0
    
    # Apply scenario
    scenario_forecast = apply_economic_scenario(
        base_forecast, scenario.value, rate_change, periods
    )
    
    # Generate forecast dates
    last_date = historical_data['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(periods)]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'cpr_forecast': scenario_forecast,
        'cpr_base': base_forecast
    })
    
    # Calculate confidence intervals
    historical_std = cpr_series.tail(12).std()
    forecast_df['lower_bound'] = forecast_df['cpr_forecast'] - 1.96 * historical_std
    forecast_df['upper_bound'] = forecast_df['cpr_forecast'] + 1.96 * historical_std
    forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
    forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(upper=1)
    
    return forecast_df, base_forecast, scenario_forecast, rate_change


@app.cell
def __(historical_data, forecast_df, go, scenario, mo):
    # Create beautiful interactive chart
    fig = go.Figure()
    
    # Historical data - solid line
    fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['cpr'],
        mode='lines+markers',
        name='Historical CPR',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=5),
        hovertemplate='Date: %{x|%Y-%m}<br>CPR: %{y:.1%}<extra></extra>'
    ))
    
    # Forecast - dashed line with different color
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['cpr_forecast'],
        mode='lines+markers',
        name=f'{scenario.value} Scenario Forecast',
        line=dict(color='#dc2626', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='Date: %{x|%Y-%m}<br>Forecast CPR: %{y:.1%}<extra></extra>'
    ))
    
    # Confidence interval - shaded area
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(220, 38, 38, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=historical_data['Date'].iloc[-1],
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    # Update layout with beautiful styling
    fig.update_layout(
        title={
            'text': f'MBS CPR Forecast - {scenario.value} Scenario',
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title='Conditional Prepayment Rate (CPR)',
            tickformat='.1%',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    # Calculate R² for display
    if len(historical_data) > 3:
        # Simple R² calculation using last 3 points
        actual = historical_data['cpr'].tail(3).values
        mean_actual = actual.mean()
        ss_tot = sum((y - mean_actual) ** 2 for y in actual)
        # For demo, using a simplified metric
        r_squared = 0.85  # Placeholder - implement proper backtesting
    else:
        r_squared = 0.0
    
# Create statistics display
@app.cell
def __(historical_data, forecast_df, go, scenario, mo, r_squared):
    # This is the NEW cell for creating KPI cards and the Chart.
    # We combine them here because they share a lot of the same data.
    
    # --- 1. Create KPI Statistics Cards ---
    
    # Calculate Trend
    last_historical = historical_data['cpr'].iloc[-1]
    first_forecast = forecast_df['cpr_forecast'].iloc[0]
    trend_direction = "Falling" if first_forecast < last_historical else "Rising"
    trend_icon = "↘️" if trend_direction == "Falling" else "↗️"
    
    # Create individual stat cards
    kpi_next_cpr = mo.stat(
        label="Next Month CPR",
        value=f"{first_forecast:.2%}",
        caption=f"vs. {last_historical:.2%} historical"
    )
    
    kpi_confidence = mo.stat(
        label="95% Confidence",
        value=f"±{ (forecast_df['upper_bound'].iloc[0] - first_forecast):.2% }",
        caption="based on historical volatility"
    )

    kpi_r_squared = mo.stat(
        label="Model R²",
        value=f"{r_squared:.3f}",
        caption="on backtest (placeholder)"
    )

    kpi_trend = mo.stat(
        label="Trend",
        value=f"{trend_direction} {trend_icon}",
        caption="vs. last historical month"
    )
    
    # Arrange stats in a row
    kpi_cards = mo.hstack([
        kpi_next_cpr, kpi_confidence, kpi_r_squared, kpi_trend
    ], justify='space-around', gap=2)


    # --- 2. Create Upgraded Interactive Chart ---

    colors = {
        'historical': '#3b82f6',  # A nice blue
        'forecast': '#16a34a',    # A distinct green
        'confidence_fill': 'rgba(22, 163, 74, 0.1)' # Light green fill
    }

    fig = go.Figure()

    # Confidence interval - MUST be added first for correct layering
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor=colors['confidence_fill'],
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Historical data - solid line
    fig.add_trace(go.Scatter(
        x=historical_data['Date'], y=historical_data['cpr'] * 100,
        mode='lines+markers', name='Historical CPR',
        line=dict(color=colors['historical'], width=2.5),
        marker=dict(size=4),
        hovertemplate='Date: %{x|%Y-%m}<br>CPR: %{y:.2f}%<extra></extra>'
    ))
    
    # Forecast - dashed line with different color
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], y=forecast_df['cpr_forecast'] * 100,
        mode='lines+markers', name='Forecast',
        line=dict(color=colors['forecast'], width=2.5, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='Date: %{x|%Y-%m}<br>Forecast: %{y:.2f}%<extra></extra>'
    ))

    # Update layout to professional standard
    fig.update_layout(
        title={
            'text': 'CPR Time Series Forecast', 'font': {'size': 20},
            'x': 0.5, 'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='CPR (%)',
        yaxis_tickformat='.0f', # Show as '20' instead of '20%'
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        height=450,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=50, r=20, t=60, b=50),
        # Use a clean template
        template='plotly_white'
    )
    
    return kpi_cards, fig, r_squared, trend_direction, first_forecast, last_historical

@app.cell
def __(historical_data, forecast_df, mo, pd):
    # Combine historical and forecast data for table view
    # Prepare historical data
    hist_display = historical_data[['Date', 'cpr', 'weighted_avg_rate', 'loan_count', 'total_upb']].copy()
    hist_display['Type'] = 'Historical'
    hist_display.rename(columns={'cpr': 'CPR'}, inplace=True)
    
    # Prepare forecast data
    forecast_display = forecast_df[['Date', 'cpr_forecast', 'lower_bound', 'upper_bound']].copy()
    forecast_display['Type'] = 'Forecast'
    forecast_display.rename(columns={'cpr_forecast': 'CPR'}, inplace=True)
    forecast_display['weighted_avg_rate'] = None
    forecast_display['loan_count'] = None
    forecast_display['total_upb'] = None
    
    # Combine and format
    combined_data = pd.concat([
        hist_display.tail(12),  # Last 12 months of historical
        forecast_display.head(12)  # First 12 months of forecast
    ])
    
    # Format the data for display
    combined_data['CPR'] = combined_data['CPR'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    combined_data['Date'] = combined_data['Date'].dt.strftime('%Y-%m')
    combined_data['total_upb'] = combined_data['total_upb'].apply(
        lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else ""
    )
    combined_data['loan_count'] = combined_data['loan_count'].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) else ""
    )
    combined_data['weighted_avg_rate'] = combined_data['weighted_avg_rate'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else ""
    )
    
    # Create bounds display for forecast rows
    for idx, row in combined_data.iterrows():
        if row['Type'] == 'Forecast' and pd.notna(row.get('lower_bound')):
            combined_data.loc[idx, 'Confidence Interval'] = f"[{row['lower_bound']:.1%}, {row['upper_bound']:.1%}]"
        else:
            combined_data.loc[idx, 'Confidence Interval'] = ""
    
    # Select columns for display
    display_columns = ['Date', 'Type', 'CPR', 'Confidence Interval', 'weighted_avg_rate', 'loan_count', 'total_upb']
    table_data = combined_data[display_columns]
    
    data_table = mo.ui.table(
        table_data,
        page_size=15,
        selection=None
    )
    
    return data_table, combined_data, table_data


@app.cell
def __(mo, fig, stats, data_table):
    # Create the final app layout
    app_view = mo.vstack([
        stats,
        mo.tabs({
            "📈 Interactive Chart": fig,
            "📊 Data Table": data_table
        })
    ])
    
    app_view
    return app_view,


if __name__ == "__main__":
    app.run()