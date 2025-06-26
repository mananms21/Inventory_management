import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Inventory Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def get_database_connection():
    """Create and return database connection"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    return conn

# Data loading function
@st.cache_data
def load_data():
    """Load and prepare data from CSV files"""
    try:
        filename = 'inventory_data.csv'
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        st.error("❌ Could not find inventory_data.csv. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Function to get available columns in a table
def get_table_columns(conn, table_name):
    """Get column names for a specific table"""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return columns
    except:
        return []

# Database setup function
def setup_database(conn, df):
    """Create database tables and insert data"""
    if df is None:
        return False
    
    try:
        # Create main inventory table from parent dataset
        df.to_sql('inventory_facts', conn, if_exists='replace', index=False)
        
        # Auto-create normalized tables from the parent dataset
        # This mimics your ERD structure
        
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
        st.error(f"Database setup error: {str(e)}")
        return False

# Analysis functions
def run_sql_query(conn, query):
    """Execute SQL query and return results"""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return pd.DataFrame()

def apply_filters(df, date_range=None, regions=None, categories=None):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply date filter
    if date_range and 'Date' in df.columns:
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df['Date']).dt.date >= start_date) & 
                (pd.to_datetime(filtered_df['Date']).dt.date <= end_date)
            ]
    
    # Apply region filter
    if regions and 'Region' in df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # Apply category filter
    if categories and 'Category' in df.columns:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    return filtered_df

def build_filtered_query(conn, base_select, date_range=None, regions=None, categories=None):
    """Build a SQL query with proper filtering based on available tables and columns"""
    
    # Check what tables exist
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    existing_tables = run_sql_query(conn, tables_query)['name'].tolist()
    
    # Check what columns exist in inventory_facts
    inventory_columns = get_table_columns(conn, 'inventory_facts')
    
    # Start building the query
    from_clause = "inventory_facts"
    where_conditions = []
    
    # Handle region filter
    if regions:
        if 'stores' in existing_tables and 'Region' in get_table_columns(conn, 'stores'):
            # Join with stores table
            from_clause = "inventory_facts i LEFT JOIN stores s ON i.Store_ID = s.Store_ID"
            region_list = "', '".join(regions)
            where_conditions.append(f"s.Region IN ('{region_list}')")
        elif 'Region' in inventory_columns:
            # Use Region directly from inventory_facts
            region_list = "', '".join(regions)
            where_conditions.append(f"Region IN ('{region_list}')")
    
    # Handle category filter
    if categories:
        if 'products' in existing_tables and 'Category' in get_table_columns(conn, 'products'):
            # Join with products table
            if 'stores' in existing_tables and regions:
                # Already have stores join
                from_clause += " LEFT JOIN products p ON i.Product_ID = p.Product_ID"
            else:
                from_clause = "inventory_facts i LEFT JOIN products p ON i.Product_ID = p.Product_ID"
            category_list = "', '".join(categories)
            where_conditions.append(f"p.Category IN ('{category_list}')")
        elif 'Category' in inventory_columns:
            # Use Category directly from inventory_facts
            category_list = "', '".join(categories)
            where_conditions.append(f"Category IN ('{category_list}')")
    
    # Handle date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if 'i.' in from_clause:  # We have joins
            where_conditions.append(f"i.Date BETWEEN '{start_date}' AND '{end_date}'")
        else:
            where_conditions.append(f"Date BETWEEN '{start_date}' AND '{end_date}'")
    
    # Build final query
    where_clause = ""
    if where_conditions:
        where_clause = " WHERE " + " AND ".join(where_conditions)
    
    # Adjust the base_select based on table structure
    if 'i.' in from_clause and 'i.' not in base_select:
        # Need to add table aliases to the select statement
        base_select = base_select.replace('Product_ID', 'i.Product_ID')
        base_select = base_select.replace('Store_ID', 'i.Store_ID')
        base_select = base_select.replace('Units_Sold', 'i.Units_Sold')
        base_select = base_select.replace('Price', 'i.Price')
        base_select = base_select.replace('Inventory_Level', 'i.Inventory_Level')
        base_select = base_select.replace('Demand_Forecast', 'i.Demand_Forecast')
    
    final_query = f"{base_select} FROM {from_clause}{where_clause}"
    return final_query

def main():
    # Header
    st.markdown('<h1 class="main-header">🏪 Inventory Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        
    if df is None:
        st.stop()
    
    # Setup database
    conn = get_database_connection()
    if not setup_database(conn, df):
        st.error("Failed to setup database")
        st.stop()
    
    # Show success message
    st.success(f"✅ Successfully loaded {len(df):,} records from inventory data")
    
    # Sidebar for navigation and filters
    st.sidebar.title("📋 Navigation & Filters")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["📊 Overview", "💰 Sales Analysis", "📦 Inventory Management", 
         "💲 Price Optimization", "🌤️ External Factors", "🔍 Advanced Analytics"]
    )
    
    # Common filters
    st.sidebar.subheader("🔧 Filters")
    
    # Date range filter
    date_range = None
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    # Region filter - only show if Region column exists
    regions = None
    if 'Region' in df.columns:
        regions = st.sidebar.multiselect(
            "Select Regions",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    
    # Category filter - only show if Category column exists
    categories = None
    if 'Category' in df.columns:
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    
    # Apply filters to dataframe
    filtered_df = apply_filters(df, date_range, regions, categories)
    
    # Show filter summary
    if len(filtered_df) != len(df):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
        st.sidebar.markdown(f"**Filter Applied:** {((len(df) - len(filtered_df)) / len(df) * 100):.1f}% reduction")
    
    # Main content based on selected page
    if page == "📊 Overview":
        show_overview(conn, filtered_df, date_range, regions, categories)
    elif page == "💰 Sales Analysis":
        show_sales_analysis(conn, filtered_df, date_range, regions, categories)
    elif page == "📦 Inventory Management":
        show_inventory_management(conn, filtered_df, date_range, regions, categories)
    elif page == "💲 Price Optimization":
        show_price_optimization(conn, filtered_df, date_range, regions, categories)
    elif page == "🌤️ External Factors":
        show_external_factors(conn, filtered_df, date_range, regions, categories)
    elif page == "🔍 Advanced Analytics":
        show_advanced_analytics(conn, filtered_df, date_range, regions, categories)

def show_overview(conn, df, date_range=None, regions=None, categories=None):
    """Display overview dashboard"""
    st.header("📊 Inventory Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        query = build_filtered_query(conn, "SELECT COUNT(DISTINCT Product_ID) as count", date_range, regions, categories)
        total_skus = run_sql_query(conn, query)
        if not total_skus.empty:
            st.metric("Total SKUs", f"{total_skus.iloc[0]['count']:,}")
    
    with col2:
        query = build_filtered_query(conn, "SELECT COUNT(DISTINCT Store_ID) as count", date_range, regions, categories)
        total_stores = run_sql_query(conn, query)
        if not total_stores.empty:
            st.metric("Total Stores", f"{total_stores.iloc[0]['count']:,}")
    
    with col3:
        if 'Units_Sold' in df.columns:
            total_sales = df['Units_Sold'].sum()
            st.metric("Total Units Sold", f"{total_sales:,}")
    
    with col4:
        if 'Price' in df.columns and 'Units_Sold' in df.columns:
            total_revenue = (df['Price'] * df['Units_Sold']).sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        if 'Category' in df.columns and 'Units_Sold' in df.columns:
            category_sales = df.groupby('Category')['Units_Sold'].sum().reset_index()
            fig = px.pie(category_sales, values='Units_Sold', names='Category', 
                        title="Sales Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Region")
        if 'Region' in df.columns and 'Units_Sold' in df.columns:
            region_sales = df.groupby('Region')['Units_Sold'].sum().reset_index()
            fig = px.bar(region_sales, x='Region', y='Units_Sold',
                        title="Sales by Region")
            st.plotly_chart(fig, use_container_width=True)
    
    # Inventory alerts
    st.subheader("🚨 Inventory Alerts")
    
    if 'Inventory_Level' in df.columns and 'Demand_Forecast' in df.columns:
        # Low stock alerts - using pandas for simplicity
        low_stock_df = df[df['Inventory_Level'] < df['Demand_Forecast']]
        
        if not low_stock_df.empty:
            st.warning(f"⚠️ {len(low_stock_df)} products are understocked")
            with st.expander("View Understocked Items"):
                st.dataframe(low_stock_df[['Product_ID', 'Store_ID', 'Inventory_Level', 'Demand_Forecast']].head(10))

def show_sales_analysis(conn, df, date_range=None, regions=None, categories=None):
    """Display sales analysis dashboard"""
    st.header("💰 Sales Performance Analysis")
    
    # Top performing products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Products by Sales")
        query = build_filtered_query(conn, "SELECT Product_ID, SUM(Units_Sold) as Total_Sales", date_range, regions, categories)
        query += " GROUP BY Product_ID ORDER BY Total_Sales DESC LIMIT 10"
        top_products = run_sql_query(conn, query)
        
        if not top_products.empty:
            fig = px.bar(top_products, x='Product_ID', y='Total_Sales',
                        title="Top 10 Products by Units Sold")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💸 Revenue by Product")
        if 'Price' in df.columns:
            revenue_query = build_filtered_query(conn, "SELECT Product_ID, SUM(Units_Sold * Price) as Revenue", date_range, regions, categories)
            revenue_query += " GROUP BY Product_ID ORDER BY Revenue DESC LIMIT 10"
            revenue_data = run_sql_query(conn, revenue_query)
            
            if not revenue_data.empty:
                fig = px.bar(revenue_data, x='Product_ID', y='Revenue',
                            title="Top 10 Products by Revenue")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Sales trends over time
    st.subheader("📈 Sales Trends Over Time")
    if 'Date' in df.columns and 'Units_Sold' in df.columns:
        daily_sales = df.groupby('Date')['Units_Sold'].sum().reset_index()
        fig = px.line(daily_sales, x='Date', y='Units_Sold',
                     title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

def show_inventory_management(conn, df, date_range=None, regions=None, categories=None):
    """Display inventory management dashboard"""
    st.header("📦 Inventory Management")
    
    # Use pandas for inventory analysis to avoid SQL complexity
    if 'Inventory_Level' in df.columns and 'Units_Sold' in df.columns:
        st.subheader("🔄 Inventory Turnover Analysis")
        
        turnover_df = df.groupby('Product_ID').agg({
            'Units_Sold': 'sum',
            'Inventory_Level': 'mean'
        }).reset_index()
        
        turnover_df['Turnover_Ratio'] = turnover_df['Units_Sold'] / turnover_df['Inventory_Level'].replace(0, np.nan)
        turnover_df = turnover_df.dropna().sort_values('Turnover_Ratio', ascending=False).head(15)
        
        if not turnover_df.empty:
            fig = px.bar(turnover_df, x='Product_ID', y='Turnover_Ratio',
                        title="Product Turnover Ratios")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def show_price_optimization(conn, df, date_range=None, regions=None, categories=None):
    """Display price optimization dashboard"""
    st.header("💲 Price Optimization Analysis")
    
    # Price elasticity analysis
    st.subheader("📊 Price vs Sales Analysis")
    
    if 'Price' in df.columns and 'Units_Sold' in df.columns:
        # Sweet spot analysis using pandas
        sweet_spot_data = df.groupby(['Product_ID', 'Price'])['Units_Sold'].mean().reset_index()
        sweet_spot_data.columns = ['Product_ID', 'Price', 'Avg_Sales']
        
        if not sweet_spot_data.empty:
            # Select a product for detailed analysis
            products = sweet_spot_data['Product_ID'].unique()
            selected_product = st.selectbox("Select Product for Price Analysis", products)
            
            product_data = sweet_spot_data[sweet_spot_data['Product_ID'] == selected_product]
            
            fig = px.scatter(product_data, x='Price', y='Avg_Sales',
                           title=f"Price vs Sales for {selected_product}")
            st.plotly_chart(fig, use_container_width=True)

def show_external_factors(conn, df, date_range=None, regions=None, categories=None):
    """Display external factors analysis"""
    st.header("🌤️ External Factors Impact")
    
    # Weather impact (if weather data exists)
    if 'Weather_Condition' in df.columns and 'Units_Sold' in df.columns:
        st.subheader("🌦️ Weather Impact on Sales")
        weather_sales = df.groupby('Weather_Condition')['Units_Sold'].mean().reset_index()
        
        fig = px.bar(weather_sales, x='Weather_Condition', y='Units_Sold',
                    title="Average Sales by Weather Condition")
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    if 'Date' in df.columns and 'Units_Sold' in df.columns:
        st.subheader("📅 Seasonal Analysis")
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        monthly_sales = df.groupby('Month')['Units_Sold'].sum().reset_index()
        
        fig = px.line(monthly_sales, x='Month', y='Units_Sold',
                     title="Monthly Sales Pattern")
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(conn, df, date_range=None, regions=None, categories=None):
    """Display advanced analytics"""
    st.header("🔍 Advanced Analytics")
    
    # Anomaly detection
    st.subheader("🚨 Sales Anomaly Detection")
    
    if 'Units_Sold' in df.columns and 'Date' in df.columns:
        # Simple anomaly detection using IQR
        Q1 = df['Units_Sold'].quantile(0.25)
        Q3 = df['Units_Sold'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = df[(df['Units_Sold'] < lower_bound) | (df['Units_Sold'] > upper_bound)]
        
        if not anomalies.empty:
            st.warning(f"⚠️ Detected {len(anomalies)} sales anomalies")
            
            fig = px.scatter(df, x='Date', y='Units_Sold', title="Sales Over Time with Anomalies")
            fig.add_scatter(x=anomalies['Date'], y=anomalies['Units_Sold'], 
                          mode='markers', marker=dict(color='red', size=10),
                          name='Anomalies')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No significant anomalies detected")
    
    # Correlation analysis
    st.subheader("📊 Correlation Analysis")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix of Numeric Variables")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
