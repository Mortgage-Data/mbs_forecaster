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
        "Crisis": 0.3
    }
    multiplier = multipliers[scenario.value]

    # This SQL query runs INSIDE DuckDB.
    # It calculates a scenario-adjusted CPR.
    # Replace 'your_table' and date/cpr columns with your actual schema.
    query = f"""
    SELECT
        reporting_date,
        AVG(pool_cpr) AS historical_cpr,
        AVG(pool_cpr) * {multiplier} AS scenario_cpr
    FROM your_table
    GROUP BY reporting_date
    ORDER BY reporting_date
    """
    # Execute the query and return the result as a pandas DataFrame
    # This is the primary pattern: Query with DuckDB, visualize with tools that use DataFrames.
    df = con.sql(query).df()
    return df

# Get the DataFrame from the query function
cpr_data = run_query()


# Cell 5: Visualize the output with an interactive chart
# This cell reruns only when `cpr_data` changes.
fig = px.line(
    cpr_data,
    x='reporting_date',
    y=['historical_cpr', 'scenario_cpr'],
    title=f"CPR Forecast under '{scenario.value}' Scenario",
    labels={'value': 'CPR', 'reporting_date': 'Date'}
)
fig

# Cell 6: Display the raw data in a table for inspection
mo.ui.table(cpr_data)