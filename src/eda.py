import pandas as pd
from pathlib import Path

def load_operations(path: str = "data") -> pd.DataFrame:
    """
    Load and combine one or more operations_daily_*.csv files.
    If `path` is a file, load that file.
    If `path` is a directory, load all matching files inside.
    """
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.glob("operations_daily_*d.csv"))
    if not files:
        raise FileNotFoundError(f"No operations_daily_*d.csv files found in {p}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        if "date" not in df.columns:
            raise ValueError(f"'date' column not found in {f}")
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_site_meta(path: str = "data/site_meta.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'site_id' not in df.columns:
        raise ValueError(f"'site_id' column not found in {path}")
    return df

def merge_and_reindex(ops_df: pd.DataFrame, meta_df: pd.DataFrame,
                      site_col='site_id', date_col='date') -> pd.DataFrame:
    # Validate site_id coverage
    missing_sites = set(ops_df[site_col]) - set(meta_df[site_col])
    if missing_sites:
        print(f"Warning: Sites {missing_sites} in operations data not found in site_meta.csv")
    
    # Merge
    df = ops_df.merge(meta_df, on=site_col, how='left')
    
    # Remove duplicates (log if any)
    dups = df.duplicated(subset=[site_col, date_col], keep=False)
    if dups.any():
        print(f"Warning: {dups.sum()} duplicate site/date rows found, keeping last")
    df = df.drop_duplicates(subset=[site_col, date_col], keep='last')
    
    # Reindex per site
    dfs = []
    for site, g in df.groupby(site_col):
        if len(g) < 14:  # Warn if too few records for lags
            print(f"Warning: Site {site} has only {len(g)} records")
        g = g.set_index(date_col).sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(full)
        g[site_col] = site
        g.index.name = date_col
        # Fill NaN for key columns
        for col in ['units_produced', 'power_kwh']:
            if col in g.columns:
                g[col] = g[col].fillna(0)  # Or ffill: g[col].fillna(method='ffill')
        # Fill metadata with defaults
        for col in meta_df.columns:
            if col != site_col and col in g.columns:
                g[col] = g[col].fillna(meta_df[col].median() if pd.api.types.is_numeric_dtype(meta_df[col]) else meta_df[col].mode()[0])
        dfs.append(g.reset_index())
    
    res = pd.concat(dfs, ignore_index=True, sort=False).sort_values([site_col, date_col])
    return res