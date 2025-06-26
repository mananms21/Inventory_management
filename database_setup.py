import sqlite3
import pandas as pd
import streamlit as st

def create_sample_data():
    """Create sample data if no dataset is provided"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Sample data parameters
    n_products = 50
    n_stores = 10
    regions = ['North', 'South', 'East', 'West', 'Central']
    categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    
    data = []
    
    for date in dates:
        for store_id in range(1, n_stores + 1):
            for product_id in range(1, min(n_products + 1, 20)):  # Limit for performance
                
                # Generate realistic data
                base_price = np.random.uniform(10, 100)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
                weather_factor = np.random.uniform(0.8, 1.2)
                
                units_sold = max(0, int(np.random.poisson(5) * seasonal_factor * weather_factor))
                inventory_level = np.random.randint(50, 200)
                demand_forecast = int(units_sold * np.random.uniform(0.8, 1.2))
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Store_ID': f'S{store_id:03d}',
                    'Product_ID': f'P{product_id:03d}',
                    'Region': regions[store_id % len(regions)],
                    'Category': categories[product_id % len(categories)],
                    'Units_Sold': units_sold,
                    'Units_Ordered': np.random.randint(0, 50) if np.random.random() < 0.3 else 0,
                    'Inventory_Level': inventory_level,
                    'Price': round(base_price * np.random.uniform(0.9, 1.1), 2),
                    'Discount': round(np.random.uniform(0, 0.3), 2),
                    'Demand_Forecast': demand_forecast,
                    'Weather_Condition': weather_conditions[np.random.randint(0, len(weather_conditions))],
                    'Holiday_Promotion': 1 if np.random.random() < 0.1 else 0,
                    'Seasonality': 'High' if date.month in [11, 12, 1] else 'Medium' if date.month in [6, 7, 8] else 'Low',
                    'Competitor_Pricing': round(base_price * np.random.uniform(0.8, 1.2), 2)
                })
    
    return pd.DataFrame(data)

def setup_database_from_csv(csv_file):
    """Setup database from uploaded CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None

def validate_data_structure(df):
    """Validate that the dataframe has required columns"""
    required_columns = ['Product_ID', 'Store_ID', 'Units_Sold']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing required columns: {missing_columns}")
        st.info("The app will work with available columns, but some features may be limited.")
    
    return df

def convert_data_types(df):
    """Convert data types for better performance"""
    # Convert date column if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['Units_Sold', 'Units_Ordered', 'Inventory_Level', 'Price', 'Discount', 'Demand_Forecast']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_database_tables(conn, df):
    """Create necessary database tables"""
    
    # Main inventory facts table
    df.to_sql('inventory_facts', conn, if_exists='replace', index=False)
    
    # Create stores table
    if 'Store_ID' in df.columns and 'Region' in df.columns:
        stores_df = df[['Store_ID', 'Region']].drop_duplicates()
        stores_df.to_sql('stores', conn, if_exists='replace', index=False)
    
    # Create products table
    if 'Product_ID' in df.columns and 'Category' in df.columns:
        products_df = df[['Product_ID', 'Category']].drop_duplicates()
        products_df.to_sql('products', conn, if_exists='replace', index=False)
    
    # Create environment facts table
    env_columns = ['Date',
