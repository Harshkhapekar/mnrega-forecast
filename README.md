MNREGA Forecast Hub (Assam)
District-Level Demand, Utilization, and Completion Forecasting

Project Summary
The MNREGA Forecast Hub is a robust, district-scale forecasting platform built to predict monthly demand, budget utilization, and completion rates for Assam's MNREGA program. Leveraging administrative data and advanced machine learning, the app produces actionable month-ahead predictions with transparent uncertainty bands to support decision-making.

Key Features
Interactive Streamlit dashboard for data exploration and forecasting

Prediction for three KPIs: Demand (Total Households Worked), Budget Utilization, Completion

Uncertainty bands (80/95%) using conformal prediction for risk-aware planning

Expanding-window Cross-Validation, ensuring realistic temporal validation

Instant CSV download of forecast results

Timestamped artifact bundles for reproducibility

Data Description
Assam districts, Financial Year 2017–2025

Grain: District × Month × Year; ~3,500 records, 25–60 columns after feature engineering

Key fields: state_name, district_name, fin_year, month, Job_Cards_Issued, Workers_Available, Persondays_Generated, Completed_Works

Targets: Total_Households_Worked, budget_utilization, completed_ratio/Completed_Works

Features: cyclical month encoding (sin/cos), lags, rolling means/medians, encoded district effects

Methodology
Model: LightGBM (Gradient-Boosted Trees) for high tabular accuracy

Benchmarks: Linear Regression, Random Forest, Gradient Boosting, XGBoost

Validation: Expanding-window CV; training on past, predicting next period

Uncertainty: Conformal prediction for honest bands

Artifacts: Each run is versioned (model, metrics, feature spec, quantiles)

Project Structure
text
mnrega-forecast/
│
├── app/
│   ├── streamlit_app.py
│   └── pages/
├── data/
│   ├── raw/
│   ├── processed/
├── models/
│   ├── artifacts/
│   ├── metrics/
├── src/
│   ├── utils/
│   ├── pipelines/
│   ├── features/
├── requirements.txt
├── README.md
└── .gitignore
How to Run
text
git clone https://github.com/<your-username>/mnrega-forecast.git
cd mnrega-forecast
python -m venv .venv
.venv\Scripts\activate   # For Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
Example Usage
Select district and time horizon in Streamlit UI

View forecasted KPIs and uncertainty bands on interactive charts

Export results to CSV

Results
LightGBM outperforms classical baselines (e.g., RMSE reduction, improved interval coverage)

Artifacts provide transparent, reproducible modeling

Band plots support resource allocation and risk mitigation

Limitations & Future Work
Accuracy depends on updated, high-quality data

Next: Improved explainability (feature importance), hierarchical models, API deployment, integration of weather/event data

Author
Harsh Khapekar | [Your Email or GitHub]

License
MIT License (see LICENSE file)
