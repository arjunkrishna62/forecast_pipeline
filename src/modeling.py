import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def forecast_site(df_site, target='units_produced'):
    df_site = df_site.set_index('date').sort_index()
    train = df_site.iloc[:-30]  # Assume enough data
    test = df_site.iloc[-30:]
    
    # Baseline: ARIMA
    model_arima = ARIMA(train[target], order=(5,1,0))  # Tune order if needed
    fit_arima = model_arima.fit()
    pred_arima = fit_arima.forecast(len(test))
    mae_base = mean_absolute_error(test[target], pred_arima)
    mape_base = mean_absolute_percentage_error(test[target], pred_arima)
    
    # Improved: Prophet
    prophet_df = train.reset_index()[['date', target]].rename(columns={'date': 'ds', target: 'y'})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=len(test) + 14)  # +14 for forecast
    pred_prophet = model_prophet.predict(future)
    pred_improved = pred_prophet['yhat'].iloc[-len(test)-14:-14]  # Test part
    mae_imp = mean_absolute_error(test[target], pred_improved)
    mape_imp = mean_absolute_percentage_error(test[target], pred_improved)
    
    # Future forecast
    future_dates = pd.date_range(df_site.index[-1] + pd.Timedelta(days=1), periods=14)
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast_baseline': fit_arima.forecast(14),
        'forecast_improved': pred_prophet['yhat'].iloc[-14:]
    })
    
    print(f'{target} - Baseline MAE/MAPE: {mae_base:.2f}/{mape_base:.2%}, Improved: {mae_imp:.2f}/{mape_imp:.2%}')
    return forecast_df

def generate_forecasts(df, output_path='outputs/'):
    forecasts_units = []
    forecasts_power = []
    sites = df['site_id'].unique()
    for site in sites:
        df_site = df[df['site_id'] == site]
        forecasts_units.append(forecast_site(df_site, 'units_produced').assign(site_id=site))
        forecasts_power.append(forecast_site(df_site, 'power_kwh').assign(site_id=site))
    pd.concat(forecasts_units).to_csv(output_path + 'forecast_units.csv', index=False)
    pd.concat(forecasts_power).to_csv(output_path + 'forecast_power.csv', index=False)