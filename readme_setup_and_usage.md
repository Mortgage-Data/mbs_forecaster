# mbs_forecaster - MBS CPR Time Series Forecasting Tool
This tool provides interactive forecasts for MBS prepayment rates.

# Usage
source venv/bin/activate
streamlit run app10.py

## Setup

1.  **Repository:**
    ```bash
    gh repo create Mortgage-Data/mbs_forecaster --public --add-readme --gitignore Python
    gh repo clone Mortgage-Data/mbs_forecaster
    <!-- or -->
    git clone [https://github.com/Mortgage-Data/mbs_forecaster.git](https://github.com/Mortgage-Data/mbs_forecaster.git)
    cd mbs_forecaster
    ```

2.  **Create Environment & Install Dependencies:**
    ```bash
    # Deactivate current environment
    deactivate

    # Remove the problematic environment
    rm -rf venv

    # Create fresh environment
    python3 -m venv venv
    source venv/bin/activate

    # Install packages in specific order to control versions
    pip install "numpy>=1.24,<2.0"
    pip install "packaging>=20,<25" 
    pip install streamlit
    pip install duckdb plotly
    pip install pmdarima
    pip install prophet
    pip install lightgbm
    ``` 
    *(Note: create a `requirements.txt` file with `pip freeze > requirements.txt`)*

3.  **Data Requirement:**
    This project requires the `mbs.db` DuckDB database file.

## Data Guidance
MBS Overview:
https://capitalmarkets.fanniemae.com/media/4271/display

Whole loans:
https://capitalmarkets.fanniemae.com/media/4426/display

Rounding:
https://capitalmarkets.fanniemae.com/media/20951/display
Due to borrower privacy considerations, we provide rounded original unpaid principal balances (UPB) for RPLs for the life
of the loan and rounded scheduled UPB for the first six months after the loan is originated or modified.
