import numpy as np
import pandas as pd

def add_date_features(df, date_col='date'):
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    return df

def add_lag_roll(df, site_col='site_id', target_cols=['units_produced','power_kwh']):
    df = df.sort_values([site_col,'date']).copy()
    for t in target_cols:
        for lag in [1,2,3,7,14]:
            df[f'{t}_lag{lag}'] = df.groupby(site_col)[t].shift(lag)
        df[f'{t}_rmean7'] = df.groupby(site_col)[t].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f'{t}_rstd7'] = df.groupby(site_col)[t].transform(lambda s: s.rolling(7, min_periods=1).std())
        df[f'{t}_pctchg1'] = df.groupby(site_col)[t].pct_change(1)
    df['kwh_per_unit'] = df['power_kwh'] / (df['units_produced'].replace(0, np.nan))
    df['is_zero_units'] = (df['units_produced'] == 0).astype(int)
    df['is_zero_power'] = (df['power_kwh'] == 0).astype(int)
    return df