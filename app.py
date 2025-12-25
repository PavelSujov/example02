import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Tuple, Optional

# Import custom modules
from data_loader import load_data, filter_data_by_date
from analysis import calculate_kpis, calculate_moving_average, get_data_overview, get_correlation_matrix, extract_yearly_data, check_missing_values
from plotting import create_time_series_plot, create_correlation_heatmap, create_histogram, create_scatter_plot, create_box_plot, create_multiple_time_series, create_yearly_trend_plot


def setup_page_config():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Анализатор Трафика Веб-сайта",
        page_icon="📊",
        layout="wide"
    )


def display_header():
    """Display the application header."""
    st.title("Анализатор Трафика Веб-сайта")
    st.markdown("---")


def setup_sidebar(df: pd.DataFrame) -> Tuple[date, date, bool]:
    """
    Set up the sidebar with interactive controls.
    
    Returns:
        Tuple of (start_date, end_date, use_smoothing)
    """
    st.sidebar.header("Параметры анализа")
    
    # File uploader for custom CSV
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите свой CSV файл",
        type=["csv"],
        help="Загрузите CSV файл с данными для анализа"
    )
    
    # Use uploaded file if provided, otherwise use default
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';')  # Try semicolon separator first
        except:
            try:
                df = pd.read_csv(uploaded_file, sep=',')  # Try comma separator
            except:
                st.sidebar.error("Ошибка при загрузке файла. Убедитесь, что файл в формате CSV.")
                df = load_data("./Doc/synthetic_sales_data.csv")  # Fallback to default
    
    # Date range selection
    date_column = 'Date' if 'Date' in df.columns else 'Дата' if 'Дата' in df.columns else None
    
    if date_column and not df.empty:
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
    else:
        # Default date range if data is not available
        min_date = date(2020, 1, 1)
        max_date = date.today()
    
    start_date = st.sidebar.date_input(
        "Начальная дата",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "Конечная дата",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Smoothing option
    use_smoothing = st.sidebar.checkbox(
        "Включить сглаживание (скользящее среднее 7 дней)",
        value=False
    )
    
    return start_date, end_date, use_smoothing


def display_kpis(df: pd.DataFrame, sessions_column: str = 'Entries quantity'):
    """Display KPI metrics."""
    if df.empty:
        st.warning("Нет данных для отображения KPI")
        return
    
    kpis = calculate_kpis(df, sessions_column)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Всего сессий",
            value=f"{kpis['total_sessions']:,}",
            help="Общее количество сессий за выбранный период"
        )
    
    with col2:
        st.metric(
            label="Среднее количество сессий в день",
            value=f"{kpis['average_daily_sessions']:.2f}",
            help="Среднее количество сессий в день за выбранный период"
        )
    
    with col3:
        st.metric(
            label="Максимальное количество сессий",
            value=f"{kpis['max_daily_sessions']:,}",
            help="Максимальное количество сессий в один день за выбранный период"
        )


def display_visualization(
    filtered_df: pd.DataFrame,
    use_smoothing: bool,
    date_column: str = 'Date',
    sessions_column: str = 'Entries quantity'
):
    """Display the main visualization."""
    if filtered_df.empty:
        st.warning("Нет данных для отображения графика")
        return
    
    # Determine the appropriate column name for sessions
    if 'Entries quantity' in filtered_df.columns:
        sessions_column = 'Entries quantity'
    elif 'Кол-во записей' in filtered_df.columns:
        sessions_column = 'Кол-во записей'
    elif 'Sessions' in filtered_df.columns:
        sessions_column = 'Sessions'
    elif 'Page Views' in filtered_df.columns:
        sessions_column = 'Page Views'
    else:
        # Use the first numeric column if no standard column names exist
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            sessions_column = numeric_cols[0]
        else:
            st.error("Не найдены числовые столбцы для визуализации")
            return
    
    # Create the plot
    fig = create_time_series_plot(
        filtered_df,
        date_column,
        sessions_column,
        title=f"График {sessions_column} по дням",
        use_moving_average=use_smoothing
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_data_table(filtered_df: pd.DataFrame):
    """Display the filtered data in a table."""
    if filtered_df.empty:
        st.warning("Нет данных для отображения")
        return
    
    st.subheader("Данные за выбранный период")
    st.dataframe(filtered_df, use_container_width=True)


def display_eda_section(df: pd.DataFrame):
    """Display Exploratory Data Analysis section."""
    st.subheader("Разведочный анализ данных (EDA)")
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_column = 'Date' if 'Date' in df.columns else 'Дата' if 'Дата' in df.columns else None
    
    if not numeric_cols:
        st.warning("Нет числовых столбцов для анализа")
        return
    
    # Tabs for different EDA components
    tab1, tab2, tab3, tab4 = st.tabs(["Обзор данных", "Распределения", "Корреляции", "Временные ряды"])
    
    with tab1:
        st.subheader("Обзор данных")
        overview = get_data_overview(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Размеры датасета**: {overview['shape'][0]} строк, {overview['shape'][1]} столбцов")
            st.write("**Типы данных:**")
            for col, dtype in overview['data_types'].items():
                st.write(f"- {col}: {dtype}")
        
        with col2:
            st.write("**Пропущенные значения:**")
            missing_values = check_missing_values(df)
            for col, count in missing_values.items():
                if count > 0:
                    st.write(f"- {col}: {count}")
            if all(count == 0 for count in missing_values.values()):
                st.write("- Нет пропущенных значений")
        
        st.write("**Описательная статистика:**")
        if overview['descriptive_stats']:
            st.dataframe(pd.DataFrame(overview['descriptive_stats']))
    
    with tab2:
        st.subheader("Однофакторный анализ")
        
        # Create histograms and boxplots for each numeric column
        for col in numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                hist_fig = create_histogram(df, col, title=f'Распределение {col}')
                st.plotly_chart(hist_fig, use_container_width=True)
            
            with col2:
                box_fig = create_box_plot(df, col, title=f'Ящичная диаграмма {col}')
                st.plotly_chart(box_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Двухфакторный анализ")
        
        if len(numeric_cols) >= 2:
            # Correlation heatmap
            corr_fig = create_correlation_heatmap(df)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Scatter plots for pairs of numeric columns
            st.subheader("Диаграммы рассеяния")
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    x_col = numeric_cols[i]
                    y_col = numeric_cols[j]
                    
                    scatter_fig = create_scatter_plot(df, x_col, y_col, title=f'Диаграмма рассеяния: {x_col} vs {y_col}')
                    st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.warning("Недостаточно числовых столбцов для корреляционного анализа")
    
    with tab4:
        st.subheader("Анализ временных рядов")
        
        if date_column:
            # Multiple time series plot
            time_series_fig = create_multiple_time_series(df, date_column, numeric_cols, title="Временные ряды")
            st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Yearly trends
            yearly_data = extract_yearly_data(df, date_column)
            if not yearly_data.empty:
                yearly_fig = create_yearly_trend_plot(yearly_data, 'Year', numeric_cols, title="Ежегодные тренды")
                st.plotly_chart(yearly_fig, use_container_width=True)
        else:
            st.warning("Не найден столбец с датой для анализа временных рядов")


def main():
    """Main application function."""
    setup_page_config()
    display_header()
    
    # File uploader for custom CSV in sidebar
    st.sidebar.header("Параметры анализа")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите свой CSV файл",
        type=["csv"],
        key="file_uploader",
        help="Загрузите CSV файл с данными для анализа"
    )
    
    # Use uploaded file if provided, otherwise use default
    if uploaded_file is not None:
        try:
            # Try to read the uploaded CSV file with different separators
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, sep=',')
            
            # Check if date column exists and convert it
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        continue
            
            st.success("Файл успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
            # Load default data as fallback
            try:
                df = load_data("./Doc/synthetic_sales_data.csv")
                if df.empty:
                    st.error("Не удалось загрузить данные из файла Doc/synthetic_sales_data.csv")
                    st.stop()
            except Exception as e2:
                st.error(f"Ошибка при загрузке данных: {e2}")
                st.stop()
    else:
        # Load default data
        try:
            df = load_data("./Doc/synthetic_sales_data.csv")
            if df.empty:
                st.error("Не удалось загрузить данные из файла Doc/synthetic_sales_data.csv")
                st.stop()
        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {e}")
            st.stop()
    
    # Try to identify and convert date column
    date_column = None
    possible_date_columns = ['Date', 'Дата', 'date', 'DATE', 'dt', 'DT', 'timestamp', 'Timestamp']
    
    for col in df.columns:
        if col in possible_date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():  # Check if conversion was successful
                    date_column = col
                    break
            except:
                continue
    
    # If no standard date column found, try to detect any date-like column
    if date_column is None:
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'dt' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():
                        date_column = col
                        break
                except:
                    continue
    
    if date_column and not df.empty and pd.api.types.is_datetime64_any_dtype(df[date_column]):
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_column])
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
    else:
        # Default date range if data is not available or date column not found
        min_date = date(2020, 1, 1)
        max_date = date.today()
    
    start_date = st.sidebar.date_input(
        "Начальная дата",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "Конечная дата",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Smoothing option
    use_smoothing = st.sidebar.checkbox(
        "Включить сглаживание (скользящее среднее 7 дней)",
        value=False
    )
    
    # Filter data by date
    if date_column:
        filtered_df = filter_data_by_date(df, start_date, end_date, date_column)
    else:
        filtered_df = df  # If no date column, use all data
    
    # Display KPIs
    st.subheader("Ключевые показатели")
    # Use the first numeric column as sessions column if available
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sessions_column = numeric_cols[0] if numeric_cols else 'Entries quantity'
    display_kpis(filtered_df, sessions_column)
    
    # Display visualization
    st.subheader("Визуализация данных")
    display_visualization(filtered_df, use_smoothing, date_column or 'Date', sessions_column)
    
    # Display data table
    display_data_table(filtered_df)
    
    # Display EDA section
    display_eda_section(df)
    
    # Add some information about the data
    with st.expander("Информация о данных"):
        st.write(f"Всего записей в выбранном диапазоне: {len(filtered_df)}")
        st.write(f"Диапазон дат: {start_date} - {end_date}")
        if not filtered_df.empty:
            st.write("Доступные столбцы:")
            st.write(list(filtered_df.columns))


if __name__ == "__main__":
    main()