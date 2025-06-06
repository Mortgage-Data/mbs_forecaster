import marimo as mo
import duckdb
import plotly.express as px
import os # Import the os library

# Cell 1: Add a title using Marimo's markdown capabilities
mo.md("# MBS CPR Time Series Forecasting Tool")

# Cell 2: Establish a connection to your DuckDB database

# Get the database path from an environment variable.
# Fall back to a default if the variable isn't set.
# os.path.expanduser handles the '~' correctly.
db_path = os.path.expanduser(
    os.getenv('MBS_DB_PATH', '~/data/mbs.db')
)

mo.md(f"Connecting to database at: `{db_path}`") # Good for debugging

# Establish the connection
con = duckdb.connect(db_path, read_only=True)

# Cell 3: Create an interactive UI element for economic scenarios
# This directly replaces a piece of our prototype JS tool's functionality.
scenario = mo.ui.dropdown(
    options=["Normal", "Recession", "Expansion", "Crisis"],
    value="Normal",
    label="Select Economic Scenario:"
)

# Display the dropdown to the app
scenario

# Cell 4: Reactively query the data based on the selected scenario
# This cell will ONLY re-run when `scenario.value` changes.
# We use DuckDB for the heavy lifting of calculation and aggregation.
@mo.rerun(on_change=[scenario])
def run_query():
    # Define the economic multipliers from your documentation
    multipliers = {
        "Normal": 1.0,
        "Recession": 0.6,
        "Expansion": 1.3,
        "Crisis": 0.3,
    }
    multiplier = multipliers[scenario.value]

    # --- Your SQL Query ---
    # We use a multi-line f-string to hold the complex query.
    # Note the two deliberate changes for development below.
    query = f"""
    SELECT
        as_of_month as Date,
        CASE WHEN seller_name = 'UNITED SHORE FINANCIAL SERVICES, LLC' THEN 'UNITED WHOLESALE MORTGAGE, LLC' ELSE seller_name END as seller,
        agency,
        number_of_borrowers,
        channel,
        fthb,
        occupancy_status,
        sum(case when loan_correction_indicator = 'pri' then 0 else 1 end ) as loan_count,
        sum(case when loan_correction_indicator = 'pri' then 0 else current_investor_loan_upb end ) as total_upb,
        SUM(current_interest_rate_pri * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_rate,
        SUM(ltv * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_ltv,
        SUM(dti * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_dti,
        SUM(credit_score * current_investor_loan_upb) / SUM(current_investor_loan_upb) as weighted_avg_credit_score,
        CASE
            WHEN SUM(prepayable_balance) > 0
            THEN 1 - POWER(1 - (SUM(unscheduled_principal_payment) / SUM(prepayable_balance)), 12)
            ELSE 0
        END as historical_cpr,
        b.pmms30,
        b.pmms30_1m_lag,
        b.pmms30_2m_lag
    FROM main.gse_sf_mbs a
    LEFT JOIN main.pmms b
    ON a.as_of_month = b.as_of_date
    WHERE is_in_bcpr3
    AND prefix = 'CL'
    AND seller_name IN ('UNITED SHORE FINANCIAL SERVICES, LLC', 'UNITED WHOLESALE MORTGAGE, LLC')
    -- MODIFICATION 1: Uncommented and activated this date filter to keep the initial dataset small and fast.
    AND as_of_month >= '2022-01-01'
    GROUP BY ALL
    ORDER BY as_of_month
    -- MODIFICATION 2: Added a LIMIT clause for rapid testing. Remove this for the full analysis.
    LIMIT 2000;
    """
    
    # Execute the query and get a DataFrame
    df = con.sql(query).df()

    # If the query returned data, apply the scenario multiplier in Python
    if not df.empty:
        df['scenario_cpr'] = df['historical_cpr'] * multiplier
    
    return df

# Get the DataFrame from the query function
cpr_data = run_query()


# Cell 5: Visualize the output with an interactive chart
# This cell reruns only when `cpr_data` changes.
fig = px.line(
    cpr_data,
    x='Date',
    y=['historical_cpr', 'scenario_cpr'],
    title=f"CPR Forecast under '{scenario.value}' Scenario",
    labels={'value': 'CPR', 'Date': 'Month', 'variable': 'CPR Type'}
)

# In a single cell, effectively combine two cells for fig and the table

mo.tabs({
    "📊 Chart View": fig,
    "🔢 Data Table": mo.ui.table(cpr_data, page_size=10)
})