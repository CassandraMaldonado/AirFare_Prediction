# ✈️ Airline Fare Prediction with Time-Series and ML

Live App: https://airfarecast-timeseriesfp.streamlit.app

## Project Overview
Repo aimed at analyzing airline pricing trends and forecasting future ticket prices using time-series and machine learning models. The goal is to provide travelers with insights that enable smarter, more cost-effective booking decisions by exploring the dynamics of fare pricing.

## Objectives
- Analyze fare trends over time to identify patterns and anomalies.

- Understand the impact of key factors like seat availability, flight length and fare class.

- Build predictive models for future ticket pricing.

- Generate booking insights to help users make informed travel decisions.

## EDA
1. Fare Trends Over Time:
Ticket prices fluctuate significantly over time, often dipping slightly before departure

Sharp spikes reflect seasonal demand and booking surges

2. Total Fare Distribution:
Most fares are under $1,000

Outliers above $8,000 are linked to premium cabin bookings

3. Correlation Analysis:
Base fare has the strongest correlation with total fare

Seat availability shows minimal impact on pricing, suggesting other factors dominate

4. Seat Availability Impact
Ticket prices remain variable regardless of how many seats are left

Availability alone doesn’t drive price behavior

5. Non-Stop vs. Connecting Flights
Non-stop flights are consistently more expensive

Connecting flights display higher pricing variability

6. Basic Economy vs. Regular Economy
Basic Economy offers more consistent, lower prices

Regular Economy has a wider spread due to flexible options and seat upgrades

7. Fare vs. Travel Distance
Longer routes generally cost more

Some short routes are unusually expensive, likely due to monopolies or last-minute pricing

## Modeling Approach
Traditional Time Series Models
ARIMA & SARIMA: Poor generalization

Prophet: Dropped due to low accuracy

- Machine Learning Models
Random Forest (Best performer)

Decision Tree

XGBoost

Deep Learning Model
RNN-LSTM: Underperformed due to pricing noise and variance

## Dataset
Source: Dil Wong (2022)

Kaggle: Flight Prices Dataset

