def clean_data(df):
    df = df.dropna(subset=['units_produced', 'power_kwh'])  # Or impute: df.fillna(method='ffill')
    # Outliers: clip
    for col in ['units_produced', 'power_kwh', 'downtime_hours']:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = np.clip(df[col], lower, upper)
    return df

def engineer_features(df):
    df = df.sort_values(['site_id', 'date'])
    # Date features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    # Lags and rolling
    for lag in [1, 7]:
        df[f'units_produced_lag{lag}'] = df.groupby('site_id')['units_produced'].shift(lag)
        df[f'power_kwh_lag{lag}'] = df.groupby('site_id')['power_kwh'].shift(lag)
    df['units_rolling7'] = df.groupby('site_id')['units_produced'].rolling(7).mean().reset_index(0, drop=True)
    # Normalize by capacity if in meta
    if 'capacity_units' in df.columns:
        df['units_normalized'] = df['units_produced'] / df['capacity_units']
    return df.fillna(0)  # Handle NaNs from shifts