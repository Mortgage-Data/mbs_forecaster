# app.py
# usage: streamlit run app.py

import streamlit as st
import duckdb

import plotly.graph_objects as go
import plotly.io as pio

# 1) Build a Template object
tmpl = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans serif", color="#111111"),
        colorway=["#667eea", "#20c997"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=80),
        height=500
    )
)

# 2) Register & set as default
pio.templates["my_streamlit"] = tmpl
pio.templates.default = "my_streamlit"


# 1) Connect to your DuckDB file (use the full path)
con = duckdb.connect('/home/gregoliven/data2/mbs/mbs.db')

st.title("MBS CPR Time Series Forecaster")


# after the connection and title
# --- seller picker ---
# 1) Fetch unique sellers
sellers = con.execute("SELECT DISTINCT seller_name FROM gse_sf_mbs ORDER BY seller_name").fetchall()
sellers = [row[0] for row in sellers]

# 2) Let user choose
sel = st.selectbox("Select Seller", sellers)

# 3) Pull the full monthly series for that seller
df_series = con.execute(f"""
    SELECT 
      DATE_TRUNC('month', as_of_month) AS month,
      AVG(unscheduled_principal_payment) AS avg_unscheduled_principal_payment
    FROM gse_sf_mbs
    WHERE seller_name = '{sel.replace("'", "''")}'
    GROUP BY 1
    ORDER BY 1
""").df()

# st.line_chart(df_series.rename(columns={'month':'index'}).set_index('index')['avg_unscheduled_principal_payment'])


# ─── Step 5: SARIMAX Forecast with statsmodels ─────────────────────────────
# --- historical chart ---
import pandas as pd
import statsmodels.api as sm

# 5.1) User picks horizon
months = st.number_input("Forecast horizon (months)", value=6, min_value=1, max_value=24)

# 5.2) Prepare the series
series = df_series['avg_unscheduled_principal_payment'].astype(float)

# 5.3) Fit a SARIMAX(1,1,1)x(1,1,1,12) — you can tweak these orders later
with st.spinner("Training SARIMAX…"):
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

# 5.4) Forecast + confidence intervals
fc = res.get_forecast(steps=months)
forecast = fc.predicted_mean.values
conf_int = fc.conf_int(alpha=0.05).values  # shape (months, 2)

# Build the future dates index
last_date = df_series['month'].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=months,
    freq='MS'
)

# 5.5) Put into a DataFrame
fc_df = pd.DataFrame({
    'forecast':        forecast,
    'lower_CI':        conf_int[:, 0],
    'upper_CI':        conf_int[:, 1],
}, index=future_dates)


# ─── KPI CARDS ────────────────────────────────────────────────────────────
#  (This must come *after* you’ve computed `forecast`, `conf_int` and `res`)

# 1) Next‐month forecast and CI width
next_month = forecast[0]
ci_width   = conf_int[0,1] - conf_int[0,0]

# 2) In‐sample R² (1 – SS_res/SS_tot)
actual = series.values
fitted = res.fittedvalues
ss_res = ((actual - fitted) ** 2).sum()
ss_tot = ((actual - actual.mean()) ** 2).sum()
r2_score = 1 - ss_res/ss_tot

# 3) Render the three metrics
col1, col2, col3 = st.columns(3)
col1.metric("Next-Month Forecast",     f"{next_month:.2f}")
col2.metric("CI Width (95%)",          f"{ci_width:.2f}")
col3.metric("In-Sample R²",            f"{r2_score:.3f}")



# 5.6) Plot it
# st.subheader(f"{months}-Month Unscheduled Principal Payment Forecast")
# st.line_chart(fc_df)

import plotly.graph_objects as go

# 1) Historical series
hist = df_series.set_index('month')['avg_unscheduled_principal_payment']

import plotly.graph_objects as go

# ─── Unified Historical + Forecast Chart w/ Markers & Grids ──────────────
hist = df_series.set_index('month')['avg_unscheduled_principal_payment']

fig = go.Figure()

# 1) Actual history (lines + dots)
fig.add_trace(go.Scatter(
    x=hist.index,
    y=hist.values,
    mode='lines+markers',
    name='Actual',
    line=dict(color='#667eea', width=2),
    marker=dict(size=6)
))

# 2) Forecast (dashed lines + dots)
fig.add_trace(go.Scatter(
    x=fc_df.index,
    y=fc_df['forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#20c997', width=2, dash='dash'),
    marker=dict(size=6)
))

# 3) Confidence band
fig.add_trace(go.Scatter(
    x=list(fc_df.index) + list(fc_df.index[::-1]),
    y=list(fc_df['upper_CI']) + list(fc_df['lower_CI'][::-1]),
    fill='toself',
    fillcolor='rgba(32, 201, 151, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    showlegend=False
))

# 4) Layout: grid lines & monthly ticks

fig.update_layout(
    template="my_streamlit",
    title=dict(
        text=f"{months}-Month Unscheduled Principal Payment Forecast",
        x=0.5,
        font=dict(size=20)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1
    ),
    xaxis=dict(
        title="Date",
        showgrid=True,    gridcolor="#eeeeee",
        tickformat="%b %Y", tickangle=-45,
        nticks=12
    ),
    yaxis=dict(
        title="Payment",
        showgrid=True,    gridcolor="#eeeeee"
    )
)



# 4) Render it
st.plotly_chart(fig, use_container_width=True)


