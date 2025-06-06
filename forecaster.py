import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import duckdb
    import plotly.express as px
    import os
    import marimo as mo
    return duckdb, mo, os, px


@app.cell
def _(mo):
    mo.md("""# MBS CPR Time Series Forecasting Tool""")
    return


@app.cell
def _(duckdb, mo, os):
    db_path = os.path.expanduser(
        os.getenv('MBS_DB_PATH', '~/data2/mbs/mbs.db')
    )
    db_path_display = mo.md(f"**Database:** `{db_path}`")
    con = duckdb.connect(db_path, read_only=True)
    return con, db_path_display


@app.cell
def _(db_path_display):
    db_path_display
    return


@app.cell
def _(mo):
    scenario = mo.ui.dropdown(
        options=["Normal", "Recession", "Expansion", "Crisis"],
        value="Normal",
        label="Select Economic Scenario:"
    )
    return (scenario,)


@app.cell
def _(scenario):
    scenario
    return


@app.cell
def _(con, scenario):
    multipliers = {
        "Normal": 1.0, "Recession": 0.6,
        "Expansion": 1.3, "Crisis": 0.3,
    }
    multiplier = multipliers[scenario.value]

    query = """
    SELECT
        as_of_month as Date,
        CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' THEN 'UNITED WHOLESALE MORTGAGE, LLC' ELSE seller_name END as seller,
        agency, channel, fthb, occupancy_status,
        sum(case when loan_correction_indicator = 'pri' then 0 else 1 end) as loan_count,
        sum(case when loan_correction_indicator = 'pri' then 0 else current_investor_loan_upb end) as total_upb,
        SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
        SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
        SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
        SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
        CASE WHEN SUM(prepayable_balance) > 0 THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12) ELSE 0 END as historical_cpr,
        b.pmms30, b.pmms30_1m_lag, b.pmms30_2m_lag
    FROM main.gse_sf_mbs a
    LEFT JOIN main.pmms b ON a.as_of_month = b.as_of_date
    WHERE is_in_bcpr3 AND prefix = 'CL' AND seller_name IN ('UNITED SHORE FINANCIAL SERVICES, LLC', 'UNITED WHOLESALE MORTGAGE, LLC')
    AND as_of_month >= '2022-01-01'
    GROUP BY ALL ORDER BY as_of_month LIMIT 2000;
    """

    cpr_data = con.sql(query).df()
    if not cpr_data.empty:
        cpr_data['scenario_cpr'] = cpr_data['historical_cpr'] * multiplier

    return (cpr_data,)


@app.cell
def _(cpr_data, mo, px, scenario):
    # Create the figure
    fig = px.line(
        cpr_data, x='Date', y=['historical_cpr', 'scenario_cpr'],
        title=f"CPR Forecast under '{scenario.value}' Scenario",
        labels={'value': 'CPR', 'Date': 'Month', 'variable': 'CPR Type'}
    )

    # Create the tabbed layout
    app_view = mo.tabs({
        "📊 Chart View": fig,
        "🔢 Data Table": mo.ui.table(cpr_data, page_size=10)
    })

    app_view
    return


if __name__ == "__main__":
    app.run()
