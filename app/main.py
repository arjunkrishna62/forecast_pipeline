import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer, pandas as pd, os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from src.eda import load_operations, load_site_meta, merge_and_reindex
from src.features import add_date_features, add_lag_roll
from src.modeling import seasonal_naive, train_direct_models, predict_from_models, evaluate_backtest
from src.anomaly import detect_downtime

app = typer.Typer()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

@ app.command()
def run():
    """
    Run the full pipeline: clean, features, baseline, improved (fast), anomalies, executive brief.
    """
    meta_path = DATA / 'site_meta.csv'
    if not meta_path.exists():
        print("Place site_meta.csv in the data/ folder and re-run.")
        raise SystemExit(1)

    # NEW: load all ops files (30d, 60d, 270d...)
    ops = load_operations(DATA)
    meta = load_site_meta(str(meta_path))
    df = merge_and_reindex(ops, meta)

    # Baseline (seasonal-naive)
    fc_units = seasonal_naive(df, target='units_produced', horizon=14)
    fc_power = seasonal_naive(df, target='power_kwh', horizon=14)
    fc_units.to_csv(OUT / 'forecast_units.csv', index=False)
    fc_power.to_csv(OUT / 'forecast_power.csv', index=False)
    print("Wrote baseline forecast CSVs.")

    # Improved (fast direct models up to origin)
    max_date = df['date'].max()
    origin = max_date - pd.Timedelta(days=14)
    print("Training fast models up to origin:", origin.date())

    feature_cols = ['site_idx','dayofweek','is_weekend','day','month',
                    'units_produced_lag1','units_produced_lag7','units_produced_lag14',
                    'power_kwh_lag1','power_kwh_lag7','power_kwh_lag14',
                    'kwh_per_unit','is_zero_units']

    df['site_idx'] = df['site_id'].astype('category').cat.codes
    # ensure feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = df.get(c.replace('site_idx','site_id'), pd.NA)

    models_u = train_direct_models(df, target='units_produced', feature_cols=feature_cols, origin_cutoff=origin)
    models_p = train_direct_models(df, target='power_kwh', feature_cols=feature_cols, origin_cutoff=origin)

    # Build backtest preds
    back_preds = []
    for site, g in df.groupby('site_id'):
        base_row = g[g['date'] <= origin].sort_values('date').tail(1)
        if base_row.empty: continue
        base = base_row.iloc[0]
        pu = predict_from_models(models_u, base, feature_cols)
        pp = predict_from_models(models_p, base, feature_cols)
        for h in range(1,15):
            back_preds.append({'site_id': site, 'date': origin + pd.Timedelta(days=h), 'pred_units': pu.get(h), 'pred_power': pp.get(h)})
    back_df = pd.DataFrame(back_preds)
    back_df.to_csv(OUT / 'backtest_predictions.csv', index=False)
    print("Wrote backtest_predictions.csv")

    # evaluation 

    print("Evaluating baseline vs improved models...")

    baseline_u = seasonal_naive(df, target='units_produced', horizon=14)
    baseline_p = seasonal_naive(df, target='power_kwh', horizon=14)

    actuals = df[df['date'].between(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=14))]
    metrics = []

    for target, baseline_fc, col in [
        ("units_produced", baseline_u, "pred_units"),
        ("power_kwh", baseline_p, "pred_power")
    ]:
        # baseline
        merged_b = baseline_fc.merge(actuals[['site_id','date',target]], on=['site_id','date'], how='inner')
        mae_b = mean_absolute_error(merged_b[target], merged_b['forecast'])
        mape_b = mean_absolute_percentage_error(merged_b[target], merged_b['forecast'])
        metrics.append({"model":"baseline","target":target,"MAE":mae_b,"MAPE":mape_b})

        # improved
        merged_i = back_df.merge(actuals[['site_id','date',target]], on=['site_id','date'], how='inner')
        mae_i = mean_absolute_error(merged_i[target], merged_i[col])
        mape_i = mean_absolute_percentage_error(merged_i[target], merged_i[col])
        metrics.append({"model":"improved","target":target,"MAE":mae_i,"MAPE":mape_i})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    print("Wrote metrics.csv with MAE/MAPE for baseline vs improved.")

    # Final forecast using the models (fast approach)
    final_preds = []
    max_date = df['date'].max()
    for site, g in df.groupby('site_id'):
        base_row = g[g['date'] <= max_date].sort_values('date').tail(1)
        if base_row.empty: continue
        base = base_row.iloc[0]
        pu = predict_from_models(models_u, base, feature_cols)
        pp = predict_from_models(models_p, base, feature_cols)
        for h in range(1,15):
            final_preds.append({'site_id': site, 'date': max_date + pd.Timedelta(days=h), 'horizon': h, 'forecast_units': pu.get(h), 'forecast_power': pp.get(h)})

    final_df = pd.DataFrame(final_preds)
    final_units = final_df[['site_id','date','forecast_units']].rename(columns={'forecast_units':'forecast'})
    final_power = final_df[['site_id','date','forecast_power']].rename(columns={'forecast_power':'forecast'})
    final_units['lower'] = final_units['forecast']
    final_units['upper'] = final_units['forecast']
    final_power['lower'] = final_power['forecast']
    final_power['upper'] = final_power['forecast']
    final_units.to_csv(OUT / 'forecast_units.csv', index=False)
    final_power.to_csv(OUT / 'forecast_power.csv', index=False)
    print("Wrote improved forecasts to outputs/")

    # Anomalies
    alerts_u = detect_downtime(df, target='units_produced')
    alerts_p = detect_downtime(df, target='power_kwh')
    alerts = pd.concat([alerts_u, alerts_p], ignore_index=True).drop_duplicates()
    alerts.to_csv(OUT / 'alerts.csv', index=False)
    print("Wrote alerts.csv")

    # Exec brief (simple PDF)
    try:
        import matplotlib.pyplot as plt
        rep_site = alerts['site_id'].value_counts().idxmax() if not alerts.empty else df['site_id'].iloc[0]
        site_series = df[df['site_id'] == rep_site].sort_values('date').set_index('date')
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.suptitle("Executive Brief — Forecast & Downtime Insights", fontsize=14, weight='bold')
        text = ("Key insights:\n- MAD-based anomaly detector flagged probable downtime events (alerts.csv).\n- Baseline = seasonal-naive; improved = direct multi-horizon models.\n\nImpact:\n- Downtime reduces units produced and raises operational risk.\n\nAutomation triggers:\n- Trigger if score < -3 AND power <15% median OR units==0. Create ticket & notify ops.")
        plt.figtext(0.05, 0.7, text, fontsize=10, va='top')
        ax = fig.add_axes([0.05, 0.35, 0.9, 0.25])
        if 'units_produced' in site_series.columns:
            window = site_series['units_produced'].last('60D')
            ax.plot(window.index, window.values)
            ax.set_title(f'Units produced — recent (site {rep_site})')
        ax.axis('off')
        fig.savefig(OUT / 'executive_brief.pdf', bbox_inches='tight')
        plt.close(fig)
        print("Wrote executive_brief.pdf")
    except Exception as e:
        print("Could not create executive brief PDF:", e)

@ app.command()
def forecast(site_id: str = None, start: str = None, end: str = None, metric: str = 'units'):
    fn = OUT / ('forecast_units.csv' if metric == 'units' else 'forecast_power.csv')
    df = pd.read_csv(fn, parse_dates=['date'])
    if site_id:
        df = df[df['site_id'] == site_id]
    if start:
        df = df[df['date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['date'] <= pd.to_datetime(end)]
    print(df.to_csv(index=False))

@ app.command()
def anomalies(site_id: str = None, start: str = None, end: str = None):
    fn = OUT / 'alerts.csv'
    df = pd.read_csv(fn, parse_dates=['date'])
    if site_id:
        df = df[df['site_id'] == site_id]
    if start:
        df = df[df['date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['date'] <= pd.to_datetime(end)]
    print(df.to_csv(index=False))

if __name__ == "__main__":
    app()