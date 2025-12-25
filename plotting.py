import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
import numpy as np


def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Time Series Plot",
    use_moving_average: bool = False,
    window: int = 7
) -> go.Figure:
    """
    Create an interactive time series plot using Plotly.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        value_column: Name of the value column to plot
        title: Title for the plot
        use_moving_average: Whether to include moving average
        window: Window size for moving average
        
    Returns:
        Plotly figure object
    """
    if df.empty or date_column not in df.columns or value_column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для отображения", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    # Create the main plot
    fig = go.Figure()
    
    # Add the original data
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[value_column],
        mode='lines+markers',
        name=value_column,
        line=dict(width=2),
        marker=dict(size=4)
    ))
    
    # Add moving average if requested
    if use_moving_average:
        moving_avg = df[value_column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=moving_avg,
            mode='lines',
            name=f'{value_column} (Скользящее среднее)',
            line=dict(width=2, color='red', dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title=value_column,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Plotly figure object
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric data for correlation",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="Correlation Heatmap")
        return fig
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Матрица корреляций между числовыми переменными",
        xaxis_title="Переменные",
        yaxis_title="Переменные"
    )
    
    return fig


def create_histogram(df: pd.DataFrame, column: str, title: str = "Histogram") -> go.Figure:
    """
    Create a histogram for a specified column.
    
    Args:
        df: Input DataFrame
        column: Column to create histogram for
        title: Title for the histogram
        
    Returns:
        Plotly figure object
    """
    if df.empty or column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    fig = px.histogram(df, x=column, title=title, marginal="box")  # Include boxplot
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Частота"
    )
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Scatter Plot"
) -> go.Figure:
    """
    Create a scatter plot for two specified columns.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis
        y_column: Column for y-axis
        title: Title for the scatter plot
        
    Returns:
        Plotly figure object
    """
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    fig = px.scatter(df, x=x_column, y=y_column, title=title)
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return fig


def create_box_plot(df: pd.DataFrame, column: str, title: str = "Box Plot") -> go.Figure:
    """
    Create a box plot for a specified column.
    
    Args:
        df: Input DataFrame
        column: Column to create box plot for
        title: Title for the box plot
        
    Returns:
        Plotly figure object
    """
    if df.empty or column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    fig = px.box(df, y=column, title=title)
    fig.update_layout(
        yaxis_title=column
    )
    
    return fig


def create_multiple_time_series(df: pd.DataFrame, date_column: str, value_columns: list, title: str = "Multiple Time Series") -> go.Figure:
    """
    Create a time series plot with multiple value columns.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        value_columns: List of value columns to plot
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if df.empty or date_column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для отображения", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    fig = go.Figure()
    
    for col in value_columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2),
                marker=dict(size=4)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title="Значения",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_yearly_trend_plot(df: pd.DataFrame, date_column: str, value_columns: list, title: str = "Yearly Trends") -> go.Figure:
    """
    Create a yearly trend plot by aggregating data by year.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column (for x-axis)
        value_columns: List of value columns to plot
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if df.empty or date_column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для отображения", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(title=title)
        return fig
    
    fig = go.Figure()
    
    for col in value_columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Год",
        yaxis_title="Значения",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig