# app.py
# usage: streamlit run app.py

import streamlit as st
import duckdb

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

st.line_chart(df_series.rename(columns={'month':'index'}).set_index('index')['avg_unscheduled_principal_payment'])

# ─── Step 5: SARIMAX Forecast with statsmodels ─────────────────────────────
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

# 5.6) Plot it
st.subheader(f"{months}-Month Unscheduled Principal Payment Forecast")
st.line_chart(fc_df)
