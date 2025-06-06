import marimo as mo

# 1. Define the application object
app = mo.App()

# 2. Define each part of our application as a cell
# The @app.cell decorator marks a function as a Marimo cell.
# The function arguments (e.g., mo, os) are the outputs of other cells.

@app.cell
def __creator__():
    # This special cell runs first to import libraries and create objects
    # that many other cells will need.
    import duckdb
    import plotly.express as px
    import os
    # We also pass 'mo' itself so other cells can use it
    import marimo as mo
    return duckdb, mo, os, px

@app.cell
def __(mo):
    # This cell just creates the title. It depends on 'mo'.
    mo.md("# MBS CPR Time Series Forecasting Tool")
    return

@app.cell
def __(duckdb, mo, os):
    # This cell connects to the database.
    db_path = os.path.expanduser(
        os.getenv('MBS_DB_PATH', '~/data/mbs.db')
    )
    mo.md(f"**Database:** `{db_path}`")
    con = duckdb.connect(db_path, read_only=True)
    return con, db_path

@app.cell
def __(mo):
    # This cell defines the interactive dropdown widget.
    scenario = mo.ui.dropdown(
        options=["Normal", "Recession", "Expansion", "Crisis"],
        value="Normal",
        label="Select Economic Scenario:"
    )
    return scenario,

@app.cell
def __(con, scenario):
    # This cell defines our query function. It depends on the DB connection
    # and the scenario dropdown to run.
    @mo.rerun(on_change=[scenario])
    def run_query():
        multipliers = {
            "Normal": 1.0, "Recession": 0.6,
            "Expansion": 1.3, "Crisis": 0.3,
        }
        multiplier = multipliers[scenario.value]

        query = f"""
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
        df = con.sql(query).df()

        if not df.empty:
            df['scenario_cpr'] = df['historical_cpr'] * multiplier
        
        return df
    return run_query,

@app.cell
def __(run_query):
    # This cell calls the query function to get the actual data.
    cpr_data = run_query()
    return cpr_data,

@app.cell
def __(cpr_data, mo, px, scenario):
    # This final cell creates the UI output. It depends on the data,
    # the scenario dropdown, and the plotting libraries.
    # We display the dropdown UI output in this cell.
    scenario_dropdown = scenario
    
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
    
    # Return all UI elements to be displayed
    return app_view, fig, scenario_dropdown


# 3. This block allows the script to be run directly
if __name__ == "__main__":
    app.run()