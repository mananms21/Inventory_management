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
    
    try:
        # Main inventory facts table
        df.to_sql('inventory_facts', conn, if_exists='replace', index=False)
        
        # Create stores table
        if 'Store_ID' in df.columns:
            store_columns = ['Store_ID']
            if 'Region' in df.columns:
                store_columns.append('Region')
            if 'Store_Name' in df.columns:
                store_columns.append('Store_Name')
            if 'City' in df.columns:
                store_columns.append('City')
            if 'State' in df.columns:
                store_columns.append('State')
                
            stores_df = df[store_columns].drop_duplicates()
            stores_df.to_sql('stores', conn, if_exists='replace', index=False)
        
        # Create products table
        if 'Product_ID' in df.columns:
            product_columns = ['Product_ID']
            if 'Category' in df.columns:
                product_columns.append('Category')
            if 'Product_Name' in df.columns:
                product_columns.append('Product_Name')
            if 'Brand' in df.columns:
                product_columns.append('Brand')
            if 'Size' in df.columns:
                product_columns.append('Size')
                
            products_df = df[product_columns].drop_duplicates()
            products_df.to_sql('products', conn, if_exists='replace', index=False)
        
        # Create environment facts table
        env_columns = ['Date', 'Store_ID']
        optional_env_columns = ['Weather_Condition', 'Holiday_Promotion', 'Seasonality', 'Competitor_Pricing', 'Temperature', 'Day_of_Week']
        
        for col in optional_env_columns:
            if col in df.columns:
                env_columns.append(col)
        
        if len(env_columns) > 2:  # More than just Date and Store_ID
            env_df = df[env_columns].drop_duplicates()
            env_df.to_sql('environment_facts', conn, if_exists='replace', index=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error creating database tables: {str(e)}")
        return False

def get_table_info(conn):
    """Get information about created tables"""
    try:
        tables = []
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in cursor.fetchall():
            table_name = table[0]
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            count = conn.execute(count_query).fetchone()[0]
            tables.append((table_name, count))
        return tables
    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")
        return []

def validate_database_structure(conn):
    """Validate that the database was created correctly"""
    try:
        # Check if main table exists
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inventory_facts';").fetchone()
        if result:
            return True
        else:
            st.error("Main inventory_facts table was not created")
            return False
    except Exception as e:
        st.error(f"Error validating database: {str(e)}")
        return False
