import pandas as pd
import streamlit as st
from typing import Optional
from datetime import datetime


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from Excel or CSV file and perform initial preprocessing.
    
    Args:
        file_path: Path to the Excel or CSV file
        
    Returns:
        DataFrame with loaded and preprocessed data
    """
    try:
        # Check file extension to determine how to load it
        if file_path.lower().endswith('.csv'):
            # For CSV files, we need to handle potential semicolon separators
            # Also need to handle potential encoding issues
            try:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try other encodings
                df = pd.read_csv(file_path, sep=';', encoding='cp1251')
        else:
            # For Excel files
            df = pd.read_excel(file_path)
        
        # Fix column names if they have encoding issues
        df.columns = df.columns.str.replace('\ufeff', '')  # Remove BOM if present
        df.columns = df.columns.str.strip() # Remove leading/trailing spaces
        
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            # Handle DD.MM.YYYY date format
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        elif 'Дата' in df.columns:
            df['Дата'] = pd.to_datetime(df['Дата'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def filter_data_by_date(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    date_column: str = 'Date'
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: Input DataFrame
        start_date: Start date for filtering
        end_date: End date for filtering
        date_column: Name of the date column
        
    Returns:
        Filtered DataFrame
    """
    if date_column in df.columns:
        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Convert start_date and end_date to pandas datetime to ensure compatibility
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        
        # Remove any rows where the date conversion failed
        df = df.dropna(subset=[date_column])
        
        mask = (df[date_column] >= start_date_pd) & (df[date_column] <= end_date_pd)
        return df.loc[mask].copy()
    else:
        # If the expected date column doesn't exist, return the original dataframe
        st.warning(f"Date column '{date_column}' not found in data.")
        return df