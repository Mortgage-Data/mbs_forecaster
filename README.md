# mbs_forecaster - MBS CPR Time Series Forecasting Tool
This tool provides interactive forecasts for MBS prepayment rates.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Mortgage-Data/mbs_forecaster.git](https://github.com/Mortgage-Data/mbs_forecaster.git)
    cd mbs_forecaster
    ```

2.  **Create Environment & Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # Install core packages first
    pip install streamlit duckdb pandas numpy plotly statsmodels scikit-learn

    # Install lightgbm separately
    pip install lightgbm

    # Install prophet (may require additional system dependencies)
    pip install prophet

    # Install pmdarima with compatibility fix if needed
    pip install pmdarima --no-cache-dir --force-reinstall
    ``` 
    *(Note: It's good practice to create a `requirements.txt` file with `pip freeze > requirements.txt`)*

3.  **Data Requirement:**
    This project requires the `mbs.db` DuckDB database file. Please place this file in a location of your choice.

4.  **Run the Application:**
    You must tell the application where to find the database by setting the `MBS_DB_PATH` environment variable.

    ```bash
    # Example for Linux/macOS
    MBS_DB_PATH='~/data/mbs.db' marimo edit forecaster.py

    # If the file is in the current directory, you can run:
    MBS_DB_PATH='mbs.db' marimo edit forecaster.py
    ```

# venv and github
<!-- Create the repo: -->
gh repo create Mortgage-Data/mbs_forecaster --public --add-readme --gitignore Python

<!-- Clone the repo from ~/code: -->
gh repo clone Mortgage-Data/mbs_forecaster

<!-- Create the venv -->
python3 -m venv venv

<!-- Start venv:
     from ~/code/mbs_forecaster: -->
source venv/bin/activate


pip install streamlit duckdb pmdarima plotly

streamlit run app.py
