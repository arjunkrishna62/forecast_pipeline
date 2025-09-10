import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
RANDOM_SEED = 42

def seasonal_naive(df, site_col='site_id', date_col='date', target='units_produced', horizon=14, lookback_days=28):
    out = []
    last_date = df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    for site, g in df.groupby(site_col):
        ref = g[g[date_col] > (g[date_col].max() - pd.Timedelta(days=lookback_days))]
        if len(ref) < 7:
            week_mean = g[target].mean()
            preds = [week_mean]*horizon
        else:
            weekday_mean = ref.groupby(ref[date_col].dt.dayofweek)[target].mean().to_dict()
            preds = [weekday_mean.get(d.dayofweek, ref[target].mean()) for d in future_dates]
        for d,p in zip(future_dates, preds):
            out.append({site_col: site, date_col: d, 'forecast': p, 'model': 'seasonal_naive', 'target': target})
    return pd.DataFrame(out)

def train_direct_models(df, target, feature_cols, site_col='site_id', date_col='date', origin_cutoff=None):
    models = {}
    df = df.sort_values([site_col, date_col]).copy()
    for h in range(1,15):
        d = df.copy()
        d['y'] = d.groupby(site_col)[target].shift(-h)
        if origin_cutoff is not None:
            d = d[d[date_col] <= origin_cutoff]
        X = d[feature_cols].copy()
        y = d['y']
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]; y = y[mask]
        if len(X) < 30:
            models[h] = None
            continue
        try:
            m = HistGradientBoostingRegressor(random_state=RANDOM_SEED, max_iter=200)
            m.fit(X, y)
            models[h] = m
        except Exception:
            r = Ridge(alpha=1.0, random_state=RANDOM_SEED)
            r.fit(X, y)
            models[h] = r
    return models

def predict_from_models(models, base_row, feature_cols):
    preds = {}
    for h, m in models.items():
        if m is None:
            preds[h] = np.nan
            continue
        x = base_row[feature_cols].values.reshape(1,-1)
        preds[h] = float(m.predict(x)[0])
    return preds

def evaluate_backtest(pred_df, obs_df, pred_col, obs_col):
    merged = pred_df.merge(obs_df[['site_id','date',obs_col]], on=['site_id','date'], how='left')
    mask = merged[pred_col].notnull() & merged[obs_col].notnull()
    if mask.sum() == 0:
        return {'mae': None, 'mape_percent': None}
    mae = mean_absolute_error(merged.loc[mask, obs_col], merged.loc[mask, pred_col])
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = (np.abs((merged.loc[mask, obs_col] - merged.loc[mask, pred_col]) / merged.loc[mask, obs_col])).mean()*100
    return {'mae': mae, 'mape_percent': mape}