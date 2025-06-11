# Project Roadmap: MBS Performance Forecasting

This roadmap outlines the development phases for building a forecasting system to predict Conditional Prepayment Rate (CPR), 3-month CPR (CPR3), and 3-month Borrower Credit Origination Rate (BCOR3) for single-family loan-level Mortgage-Backed Securities (MBS) performance data.

## Phase 1: Local CPU-Based Development and Streamlit Delivery
**Objective**: Develop a prototype using efficient, CPU-friendly models to predict loan-level CPR, CPR3, and BCOR3, with group-level aggregation (e.g., by seller or cohort), delivered via a local Streamlit application.

- **Models**:
  - **LightGBM**: Leverage gradient boosting for regression to predict loan-level rates, enhanced with engineered features (e.g., `lag_cpr_1`, `lag_cpr_3`, `rolling_cpr_avg_3`, `lag_pmms_1`, `loan_age`). Aggregate predictions for group-level outputs.
  - **NeuralProphet**: Use as an interpretable time series model for aggregated group-level rates (e.g., monthly CPR by seller), incorporating external regressors (e.g., PMMS, HPI).
  From the paper:
  2.3.4. Learning Rate
    As we do not want to require users to be experts in machine learning, we make the learning
    rate, an essential hyperparameter for any NN, optional. We achieve this by relying on a simple 
    but reasonably effective approach to estimate a learning rate, introduced as a learning rate range
    test in Smith (2017).
    A learning rate range test is executed for 100 + log10(10 + T) ∗ 50) iterations, starting at
    η = 1e − 7, ending at η = 1e + 2 with the configured batch-size. After each iteration, the learning
    rate is increased exponentially until the final learning rate is reached in the last iteration. Excluding
    the first 10 and last 5 iterations, the steepest learning rate is defined as the resulting learning rate.
    The steepest learning rate is found by selecting the learning rate at the position which maximizes
    the negative gradient of the losses.
    In order to increase reliability, we perform the the test three times and take the log10-mean of
    the three runs.

- **Development Environment**:
  - **Hardware**: Local Intel i9 Linux box with 128 GB RAM (no GPU).
  - **Data Processing**: DuckDB for efficient SQL-based feature engineering (lags, rolling averages).
  - **Feature Engineering**: Compute loan-level lags and rolling averages for targets (CPR, CPR3, BCOR3) and predictors (PMMS, HPI, unemployment rate).
- **Delivery**:
  - **Streamlit App**: Interactive web interface for visualizing loan-level and group-level predictions, feature importance, and model performance metrics (MSE, MAE).
  - **Features**: Default to a single prediction relying on an ensemble of LightGBM and NeuralProphet. Tabs for individual model selection (LightGBM vs. NeuralProphet) for testing, and, as appropriate, prediction plots, and feature exploration (e.g., lagged PMMS vs. CPR).
- **Evaluation**:
  - Metrics: MSE, MAE, RMSE for loan-level and group-level predictions.
  - Compare LightGBM and NeuralProphet against baseline (Auto-ARIMA).
- **Timeline**: 4–6 weeks for prototype development, testing, and refinement.

## Phase 2: Cloud-Based GPU/Transformer Development
**Objective**: Transition to advanced, GPU-accelerated transformer-based models in the cloud for improved accuracy and scalability, integrating with Snowflake Cortex AI.

- **Models**:
  - **Temporal Fusion Transformer (TFT)**: Implement for multi-horizon forecasting, leveraging loan-level and macroeconomic features with attention mechanisms for interpretability.
  - **Optional Models**: Explore other transformer-based approaches (e.g., PatchTST, TimesFM) or probabilistic models (e.g., TimeGrad) if TFT performance is insufficient.
- **Development Environment**:
  - **Cloud Platform**: Snowflake for data storage and processing, with potential GPU instances via Snowpark Container Services or external clouds (e.g., AWS EC2, Google Cloud).
  - **Data Processing**: Snowflake SQL/Snowpark for feature engineering, reusing DuckDB logic (e.g., lags, rolling averages).
  - **Compute**: GPU-based training for transformer models to handle large-scale loan-level data.
- **Delivery**:
  - **Streamlit-in-Snowflake**: Extend Phase 1 Streamlit app to run natively in Snowflake, displaying TFT predictions and uncertainty estimates.
  - **Integration**: Use Snowflake Cortex AI Forecasting as a baseline or complement to TFT, with Snowpark for custom model deployment.
- **Evaluation**:
  - Metrics: MSE, MAE, CRPS (for probabilistic models) for loan-level and group-level predictions.
  - Compare TFT and optional models against Phase 1 LightGBM/NeuralProphet baselines.
- **Timeline**: 6–8 weeks for cloud setup, model implementation, and integration, following Phase 1 completion.

## Key Considerations
- **Data Privacy**: Ensure compliance with MBS data regulations using Snowflake’s governance features in Phase 2.
- **Scalability**: Phase 1 focuses on local prototyping; Phase 2 leverages cloud compute for large-scale data.
- **Transition**: Reuse DuckDB feature engineering logic in Snowflake to streamline Phase 2 development.
- **Experimentation**: Test lagging CPR components (unscheduled_principal_payment, prepayable_balance) in Phase 1 if needed, and explore additional lags (e.g., Lag2) based on feature importance.
``s