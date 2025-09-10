import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    features = ['downtime_hours', 'units_produced', 'power_kwh']  # Add more
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[features])
    anomalies = df[df['anomaly_score'] == -1].copy()
    anomalies['reason'] = 'High downtime anomaly'  # Add logic for interpretability, e.g., if downtime > 3*mean
    anomalies[['date', 'site_id', 'anomaly_score', 'reason']].to_csv('outputs/alerts.csv', index=False)
    return anomalies

def detect_downtime(df, site_col='site_id', date_col='date', target='units_produced', power_col='power_kwh',
                    window_median=7, mad_window=60, score_thresh=-3, low_power_frac=0.15):
    df = df.sort_values([site_col, date_col]).copy()
    alerts = []
    for site, g in df.groupby(site_col):
        g = g.reset_index(drop=True)
        g['expected'] = g[target].rolling(window_median, min_periods=1).median()
        g['residual'] = g[target] - g['expected']
        g['mad'] = g['residual'].rolling(mad_window, min_periods=3).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        g['scale'] = g['mad'] * 1.4826
        g['score'] = g['residual'] / g['scale'].replace(0, np.nan)
        g['median_power_30d'] = g[power_col].rolling(30, min_periods=1).median()
        cond = (g['score'] < score_thresh) & ((g[power_col] < low_power_frac * g['median_power_30d']) | (g[target] == 0))
        for _, row in g[cond].iterrows():
            alerts.append({
                site_col: site,
                date_col: row[date_col],
                'metric': target,
                'observed': row[target],
                'expected': row['expected'],
                'residual': row['residual'],
                'score': row['score'],
                'anomaly_type': 'downtime',
                'rule_triggered': 'score_and_low_power_or_zero_units',
                'explanation': f"Score={row['score']:.2f}; power={row[power_col]}; expected={row['expected']:.1f}"
            })
    return pd.DataFrame(alerts)