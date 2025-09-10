import pandas as pd
from pathlib import Path

def load_operations(path: str):
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_site_meta(path: str):
    return pd.read_csv(path)

def merge_and_reindex(ops_df: pd.DataFrame, meta_df: pd.DataFrame,
                      site_col='site_id', date_col='date') -> pd.DataFrame:
    df = ops_df.merge(meta_df, on=site_col, how='left')
    dfs = []
    for site, g in df.groupby(site_col):
        g = g.set_index(date_col).sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(full)
        g[site_col] = site
        g.index.name = date_col
        dfs.append(g.reset_index())
    res = pd.concat(dfs, ignore_index=True, sort=False).sort_values([site_col, date_col])
    return res