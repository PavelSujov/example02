import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from datetime import datetime


def calculate_kpis(df: pd.DataFrame, sessions_column: str = 'Entries quantity') -> Dict[str, Any]:
    """
    Calculate key performance indicators from the data.
    
    Args:
        df: Input DataFrame
        sessions_column: Name of the column representing sessions
        
    Returns:
        Dictionary with KPI values
    """
    if df.empty:
        return {
            'total_sessions': 0,
            'average_daily_sessions': 0.0,
            'max_daily_sessions': 0,
            'date_of_max_sessions': None
        }
    
    if sessions_column in df.columns:
        total_sessions = int(df[sessions_column].sum())
        average_daily_sessions = float(df[sessions_column].mean())
        max_daily_sessions = int(df[sessions_column].max())
        
        # Find the date of max sessions
        max_sessions_idx = df[sessions_column].idxmax()
        date_of_max_sessions = df.loc[max_sessions_idx, 'Date' if 'Date' in df.columns else 'Дата']
    else:
        # If the expected column doesn't exist, try to find any numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Use the first numeric column as sessions
            sessions_column = numeric_cols[0]
            total_sessions = int(df[sessions_column].sum())
            average_daily_sessions = float(df[sessions_column].mean())
            max_daily_sessions = int(df[sessions_column].max())
            
            max_sessions_idx = df[sessions_column].idxmax()
            date_column = 'Date' if 'Date' in df.columns else 'Дата' if 'Дата' in df.columns else df.columns[0]
            date_of_max_sessions = df.loc[max_sessions_idx, date_column]
        else:
            # If no numeric columns exist, return zeros
            total_sessions = 0
            average_daily_sessions = 0.0
            max_daily_sessions = 0
            date_of_max_sessions = None
    
    return {
        'total_sessions': total_sessions,
        'average_daily_sessions': average_daily_sessions,
        'max_daily_sessions': max_daily_sessions,
        'date_of_max_sessions': date_of_max_sessions
    }


def calculate_moving_average(df: pd.DataFrame, column: str, window: int = 7) -> pd.Series:
    """
    Calculate moving average for a specified column.
    
    Args:
        df: Input DataFrame
        column: Column to calculate moving average for
        window: Window size for moving average
        
    Returns:
        Series with moving average values
    """
    if column in df.columns:
        return df[column].rolling(window=window, min_periods=1).mean()
    else:
        # If the specified column doesn't exist, return a series of zeros
        return pd.Series([0] * len(df), index=df.index)


def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 2.0) -> pd.Series:
    """
    Detect anomalies using z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to detect anomalies in
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Boolean Series indicating anomalies
    """
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    data = df[column]
    z_scores = (data - data.mean()) / data.std()
    return abs(z_scores) > threshold


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for all numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    
    stats = {}
    for col in numeric_df.columns:
        stats[col] = {
            'mean': float(numeric_df[col].mean()),
            'median': float(numeric_df[col].median()),
            'std': float(numeric_df[col].std()),
            'min': float(numeric_df[col].min()),
            'max': float(numeric_df[col].max()),
            'q25': float(numeric_df[col].quantile(0.25)),
            'q75': float(numeric_df[col].quantile(0.75))
        }
    return stats


def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get an overview of the dataset similar to df.info() and df.describe().
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset overview
    """
    if df.empty:
        return {
            'shape': (0, 0),
            'columns': [],
            'missing_values': {},
            'data_types': {},
            'descriptive_stats': {}
        }
    
    overview = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'descriptive_stats': df.describe().to_dict()
    }
    
    return overview


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column names and their missing value counts
    """
    return df.isnull().sum().to_dict()


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def extract_yearly_data(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract yearly aggregated data from the DataFrame.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with yearly aggregated data
    """
    if df.empty or date_column not in df.columns:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Extract year from date column
    df_copy['Year'] = df_copy[date_column].dt.year
    
    # Get numeric columns for aggregation (excluding the Year column if it exists already)
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'Year' from numeric columns if it's there
    if 'Year' in numeric_cols:
        numeric_cols.remove('Year')
    
    # Group by year and sum the values
    yearly_data = df_copy.groupby('Year')[numeric_cols].sum().reset_index()
    
    return yearly_data