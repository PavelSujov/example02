import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from analysis import (
    calculate_kpis, 
    calculate_moving_average, 
    detect_anomalies, 
    get_summary_statistics,
    get_data_overview,
    check_missing_values,
    get_correlation_matrix,
    extract_yearly_data
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Entries quantity': [100, 120, 90, 110, 130, 80, 140, 105, 95, 125],
        'Product A': [50, 60, 45, 55, 65, 40, 70, 52, 48, 62],
        'Product B': [30, 40, 25, 35, 45, 20, 50, 33, 28, 43],
        'Product C': [20, 25, 20, 30, 20, 15, 25, 22, 18, 30]
    })
    return df


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for testing."""
    return pd.DataFrame()


@pytest.fixture
def dataframe_with_nans():
    """Create a DataFrame with some NaN values for testing."""
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'Entries quantity': [10, np.nan, 90, 110, np.nan],
        'Product A': [50, 60, np.nan, 55, 65]
    })
    return df


class TestCalculateKPIs:
    """Test cases for calculate_kpis function."""
    
    def test_calculate_kpis_with_valid_data(self, sample_dataframe):
        """Test calculate_kpis with valid data."""
        result = calculate_kpis(sample_dataframe, 'Entries quantity')
        
        assert result['total_sessions'] == 1095  # Sum of entries
        assert result['average_daily_sessions'] == 109.5  # Mean of entries
        assert result['max_daily_sessions'] == 140  # Max of entries
        assert result['date_of_max_sessions'] == pd.Timestamp('2023-01-07')
    
    def test_calculate_kpis_with_empty_dataframe(self, empty_dataframe):
        """Test calculate_kpis with empty DataFrame."""
        result = calculate_kpis(empty_dataframe, 'Entries quantity')
        
        expected = {
            'total_sessions': 0,
            'average_daily_sessions': 0.0,
            'max_daily_sessions': 0,
            'date_of_max_sessions': None
        }
        assert result == expected
    
    def test_calculate_kpis_with_missing_column(self, sample_dataframe):
        """Test calculate_kpis when specified column doesn't exist."""
        result = calculate_kpis(sample_dataframe, 'NonExistentColumn')
        
        # Should use the first numeric column available (Entries quantity)
        assert result['total_sessions'] == 1095  # Sum of Entries quantity
        assert result['average_daily_sessions'] == 109.5  # Mean of Entries quantity
        assert result['max_daily_sessions'] == 140  # Max of Entries quantity
    
    def test_calculate_kpis_with_nans(self, dataframe_with_nans):
        """Test calculate_kpis with NaN values."""
        result = calculate_kpis(dataframe_with_nans, 'Entries quantity')
        
        # NaN values should be ignored in calculations
        assert result['total_sessions'] == 210  # 10 + 90 + 110 = 210 (excluding NaNs)
        assert result['average_daily_sessions'] == 70  # Average of non-NaN values (210/3)
        assert result['max_daily_sessions'] == 110  # Max of non-NaN values


class TestCalculateMovingAverage:
    """Test cases for calculate_moving_average function."""
    
    def test_calculate_moving_average_with_valid_column(self, sample_dataframe):
        """Test calculate_moving_average with valid column."""
        result = calculate_moving_average(sample_dataframe, 'Entries quantity', window=3)
        
        # Check that result has the same length as input
        assert len(result) == len(sample_dataframe)
        
        # Check first few values (with minimum periods)
        assert result.iloc[0] == 100 # First value
        assert result.iloc[1] == 110 # Average of 100, 120
    
    def test_calculate_moving_average_with_invalid_column(self, sample_dataframe):
        """Test calculate_moving_average with non-existent column."""
        result = calculate_moving_average(sample_dataframe, 'NonExistentColumn')
        
        # Should return series of zeros with same index
        expected = pd.Series([0] * len(sample_dataframe), index=sample_dataframe.index)
        pd.testing.assert_series_equal(result, expected)
    
    def test_calculate_moving_average_with_different_window(self, sample_dataframe):
        """Test calculate_moving_average with different window sizes."""
        result = calculate_moving_average(sample_dataframe, 'Entries quantity', window=5)
        
        assert len(result) == len(sample_dataframe)


class TestDetectAnomalies:
    """Test cases for detect_anomalies function."""
    
    def test_detect_anomalies_with_valid_column(self, sample_dataframe):
        """Test detect_anomalies with valid column."""
        result = detect_anomalies(sample_dataframe, 'Entries quantity', threshold=1.0)
        
        # Check that result is a boolean series with same length
        assert len(result) == len(sample_dataframe)
        assert result.dtype == bool
    
    def test_detect_anomalies_with_invalid_column(self, sample_dataframe):
        """Test detect_anomalies with non-existent column."""
        result = detect_anomalies(sample_dataframe, 'NonExistentColumn')
        
        # Should return series of False values
        expected = pd.Series([False] * len(sample_dataframe), index=sample_dataframe.index)
        pd.testing.assert_series_equal(result, expected)
    
    def test_detect_anomalies_with_extreme_values(self):
        """Test detect_anomalies with extreme values."""
        df = pd.DataFrame({'values': [1, 2, 3, 4, 100]})  # 100 is anomaly
        result = detect_anomalies(df, 'values', threshold=1.0)
        
        # The last value (100) should be detected as anomaly
        assert result.iloc[-1] == True


class TestGetSummaryStatistics:
    """Test cases for get_summary_statistics function."""
    
    def test_get_summary_statistics_with_numeric_data(self, sample_dataframe):
        """Test get_summary_statistics with numeric data."""
        result = get_summary_statistics(sample_dataframe)
        
        # Should contain statistics for all numeric columns
        assert 'Entries quantity' in result
        assert 'Product A' in result
        assert 'Product B' in result
        assert 'Product C' in result
        
        # Check specific values for one column
        entries_stats = result['Entries quantity']
        assert entries_stats['mean'] == 109.5
        assert entries_stats['std'] == pytest.approx(19.0685896466181, abs=0.001)
        assert entries_stats['min'] == 80
        assert entries_stats['max'] == 140
    
    def test_get_summary_statistics_with_empty_dataframe(self, empty_dataframe):
        """Test get_summary_statistics with empty DataFrame."""
        result = get_summary_statistics(empty_dataframe)
        
        assert result == {}
    
    def test_get_summary_statistics_with_no_numeric_columns(self):
        """Test get_summary_statistics with no numeric columns."""
        df = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'date_col': pd.date_range(start='2023-01-01', periods=3)
        })
        result = get_summary_statistics(df)
        
        assert result == {}


class TestGetDataOverview:
    """Test cases for get_data_overview function."""
    
    def test_get_data_overview_with_sample_data(self, sample_dataframe):
        """Test get_data_overview with sample data."""
        result = get_data_overview(sample_dataframe)
        
        assert result['shape'] == (10, 5)  # 10 rows, 5 columns
        assert 'Date' in result['columns']
        assert 'Entries quantity' in result['columns']
        assert 'Product A' in result['columns']
        assert 'Product B' in result['columns']
        assert 'Product C' in result['columns']
        
        # Check that there are no missing values
        for count in result['missing_values'].values():
            assert count == 0
    
    def test_get_data_overview_with_empty_dataframe(self, empty_dataframe):
        """Test get_data_overview with empty DataFrame."""
        result = get_data_overview(empty_dataframe)
        
        assert result['shape'] == (0, 0)
        assert result['columns'] == []
        assert result['missing_values'] == {}
        assert result['data_types'] == {}
        assert result['descriptive_stats'] == {}
    
    def test_get_data_overview_with_nans(self, dataframe_with_nans):
        """Test get_data_overview with NaN values."""
        result = get_data_overview(dataframe_with_nans)
        
        # Check missing values count
        assert result['missing_values']['Entries quantity'] == 2
        assert result['missing_values']['Product A'] == 1


class TestCheckMissingValues:
    """Test cases for check_missing_values function."""
    
    def test_check_missing_values_with_nans(self, dataframe_with_nans):
        """Test check_missing_values with NaN values."""
        result = check_missing_values(dataframe_with_nans)
        
        assert result['Entries quantity'] == 2
        assert result['Product A'] == 1
        assert result['Date'] == 0
    
    def test_check_missing_values_with_no_nans(self, sample_dataframe):
        """Test check_missing_values with no NaN values."""
        result = check_missing_values(sample_dataframe)
        
        for count in result.values():
            assert count == 0


class TestGetCorrelationMatrix:
    """Test cases for get_correlation_matrix function."""
    
    def test_get_correlation_matrix_with_numeric_data(self, sample_dataframe):
        """Test get_correlation_matrix with numeric data."""
        result = get_correlation_matrix(sample_dataframe)
        
        # Check that it's a correlation matrix (square matrix with 1s on diagonal)
        assert result.shape[0] == result.shape[1]  # Square matrix
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
        assert result.shape[0] == len(numeric_cols)  # Correct size
        
        # Diagonal values should be 1.0 (perfect correlation with self)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0, abs=0.001)
    
    def test_get_correlation_matrix_with_empty_dataframe(self, empty_dataframe):
        """Test get_correlation_matrix with empty DataFrame."""
        result = get_correlation_matrix(empty_dataframe)
        
        # Should return an empty DataFrame
        assert result.empty


class TestExtractYearlyData:
    """Test cases for extract_yearly_data function."""
    
    def test_extract_yearly_data_with_date_column(self):
        """Test extract_yearly_data with date column."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2020-01-01', '2020-06-15', '2021-01-01', '2021-06-15']),
            'Value1': [10, 20, 30, 40],
            'Value2': [5, 15, 25, 35]
        })
        
        result = extract_yearly_data(df, 'Date')
        
        # Should have years as index and aggregated values
        assert 2020 in result['Year'].values
        assert 2021 in result['Year'].values
        
        # Values for 2020 should be sums of first two rows
        year_2020 = result[result['Year'] == 2020]
        assert year_2020['Value1'].iloc[0] == 30  # 10 + 20
        assert year_2020['Value2'].iloc[0] == 20  # 5 + 15
    
    def test_extract_yearly_data_with_empty_dataframe(self, empty_dataframe):
        """Test extract_yearly_data with empty DataFrame."""
        result = extract_yearly_data(empty_dataframe, 'Date')
        
        assert result.empty
    
    def test_extract_yearly_data_with_invalid_date_column(self, sample_dataframe):
        """Test extract_yearly_data with invalid date column."""
        result = extract_yearly_data(sample_dataframe, 'NonExistentDateColumn')
        
        # Should return empty DataFrame
        assert result.empty