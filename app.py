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
        # Try common filename patterns for your dataset
        possible_files = [
            'inventory_data.csv',
            'urban_inventory.csv', 
            'inventory_dataset.csv',
            'main_dataset.csv',
            'parent_dataset.csv'
        ]
        
        df = None
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                st.success(f"✅ Successfully loaded data from {filename}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            st.error("Dataset file not found. Please ensure your CSV file is uploaded with one of these names:")
            st.write("• inventory_data.csv")
            st.write("• urban_inventory.csv") 
            st.write("• inventory_dataset.csv")
            st.write("• main_dataset.csv")
            st.write("• parent_dataset.csv")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

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
        
        # Create inventory facts table (this might be the same as main table or a subset)
        inventory_columns = ['Date', 'Store_ID', 'Product_ID']
        optional_inv_columns = ['Inventory_Level', 'Units_Sold', 'Units_Ordered', 'Price', 'Discount', 'Demand_Forecast', 'Cost', 'Lead_Time']
        
        for col in optional_inv_columns:
            if col in df.columns:
                inventory_columns.append(col)
                
        # If we have these core columns, create the inventory_facts table
        if all(col in df.columns for col in ['Date', 'Store_ID', 'Product_ID']):
            inventory_df = df[inventory_columns]
            inventory_df.to_sql('inventory_facts', conn, if_exists='replace', index=False)
        
        st.info(f"✅ Database setup complete. Created tables from {len(df)} records.")
        
        # Show table summary
        tables_created = []
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in cursor.fetchall():
            table_name = table[0]
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            count = conn.execute(count_query).fetchone()[0]
            tables_created.append(f"• {table_name}: {count:,} records")
        
        with st.expander("📋 Database Tables Created"):
            for table_info in tables_created:
                st.write(table_info)
        
        return True
    except Exception as e:
        st.error(f"Database setup error: {str(e)}")
        st.write("Available columns in your dataset:")
        st.write(list(df.columns))
        return False

# Analysis functions
def run_sql_query(conn, query):
    """Execute SQL query and return results"""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return pd.DataFrame()

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
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    
    # Region filter
    if 'Region' in df.columns:
        regions = st.sidebar.multiselect(
            "Select Regions",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    
    # Category filter
    if 'Category' in df.columns:
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    
    # Main content based on selected page
    if page == "📊 Overview":
        show_overview(conn, df)
    elif page == "💰 Sales Analysis":
        show_sales_analysis(conn, df)
    elif page == "📦 Inventory Management":
        show_inventory_management(conn, df)
    elif page == "💲 Price Optimization":
        show_price_optimization(conn, df)
    elif page == "🌤️ External Factors":
        show_external_factors(conn, df)
    elif page == "🔍 Advanced Analytics":
        show_advanced_analytics(conn, df)

def show_overview(conn, df):
    """Display overview dashboard"""
    st.header("📊 Inventory Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_skus = run_sql_query(conn, "SELECT COUNT(DISTINCT Product_ID) as count FROM inventory_facts")
        if not total_skus.empty:
            st.metric("Total SKUs", f"{total_skus.iloc[0]['count']:,}")
    
    with col2:
        total_stores = run_sql_query(conn, "SELECT COUNT(DISTINCT Store_ID) as count FROM inventory_facts")
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
        # Low stock alerts
        low_stock = run_sql_query(conn, """
            SELECT Product_ID, Store_ID, Inventory_Level, Demand_Forecast
            FROM inventory_facts 
            WHERE Inventory_Level < Demand_Forecast
            ORDER BY (Demand_Forecast - Inventory_Level) DESC
            LIMIT 10
        """)
        
        if not low_stock.empty:
            st.warning(f"⚠️ {len(low_stock)} products are understocked")
            with st.expander("View Understocked Items"):
                st.dataframe(low_stock)

def show_sales_analysis(conn, df):
    """Display sales analysis dashboard"""
    st.header("💰 Sales Performance Analysis")
    
    # Top performing products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Products by Sales")
        top_products = run_sql_query(conn, """
            SELECT Product_ID, SUM(Units_Sold) as Total_Sales
            FROM inventory_facts
            GROUP BY Product_ID
            ORDER BY Total_Sales DESC
            LIMIT 10
        """)
        
        if not top_products.empty:
            fig = px.bar(top_products, x='Product_ID', y='Total_Sales',
                        title="Top 10 Products by Units Sold")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💸 Revenue by Product")
        revenue_query = """
            SELECT Product_ID, SUM(Units_Sold * Price) as Revenue
            FROM inventory_facts
            GROUP BY Product_ID
            ORDER BY Revenue DESC
            LIMIT 10
        """
        revenue_data = run_sql_query(conn, revenue_query)
        
        if not revenue_data.empty:
            fig = px.bar(revenue_data, x='Product_ID', y='Revenue',
                        title="Top 10 Products by Revenue")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Sales trends over time
    st.subheader("📈 Sales Trends Over Time")
    if 'Date' in df.columns and 'Units_Sold' in df.columns:
        daily_sales = df.groupby('Date')['Units_Sold'].sum().reset_index()
        fig = px.line(daily_sales, x='Date', y='Units_Sold',
                     title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

def show_inventory_management(conn, df):
    """Display inventory management dashboard"""
    st.header("📦 Inventory Management")
    
    # Inventory turnover analysis
    st.subheader("🔄 Inventory Turnover Analysis")
    turnover_query = """
        SELECT Product_ID,
               ROUND(SUM(Units_Sold) / NULLIF(AVG(Inventory_Level), 0), 2) as Turnover_Ratio
        FROM inventory_facts
        GROUP BY Product_ID
        ORDER BY Turnover_Ratio DESC
        LIMIT 15
    """
    turnover_data = run_sql_query(conn, turnover_query)
    
    if not turnover_data.empty:
        fig = px.bar(turnover_data, x='Product_ID', y='Turnover_Ratio',
                    title="Product Turnover Ratios")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reorder recommendations
    st.subheader("🔔 Reorder Recommendations")
    reorder_query = """
        SELECT Store_ID, Product_ID, 
               ROUND(AVG(Units_Sold) * 7, 2) as Reorder_Point,
               MAX(Inventory_Level) as Current_Inventory
        FROM inventory_facts
        GROUP BY Store_ID, Product_ID
        HAVING Current_Inventory <= Reorder_Point
        ORDER BY (Reorder_Point - Current_Inventory) DESC
        LIMIT 20
    """
    reorder_data = run_sql_query(conn, reorder_query)
    
    if not reorder_data.empty:
        st.warning(f"⚠️ {len(reorder_data)} products need reordering")
        st.dataframe(reorder_data)
    else:
        st.success("✅ All products are adequately stocked")

def show_price_optimization(conn, df):
    """Display price optimization dashboard"""
    st.header("💲 Price Optimization Analysis")
    
    # Price elasticity analysis
    st.subheader("📊 Price vs Sales Analysis")
    
    if 'Price' in df.columns and 'Units_Sold' in df.columns:
        # Sweet spot analysis
        sweet_spot_query = """
            SELECT Product_ID, Price, AVG(Units_Sold) as Avg_Sales
            FROM inventory_facts
            GROUP BY Product_ID, Price
            ORDER BY Product_ID, Avg_Sales DESC
        """
        sweet_spot_data = run_sql_query(conn, sweet_spot_query)
        
        if not sweet_spot_data.empty:
            # Select a product for detailed analysis
            products = sweet_spot_data['Product_ID'].unique()
            selected_product = st.selectbox("Select Product for Price Analysis", products)
            
            product_data = sweet_spot_data[sweet_spot_data['Product_ID'] == selected_product]
            
            fig = px.scatter(product_data, x='Price', y='Avg_Sales',
                           title=f"Price vs Sales for {selected_product}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Discount effectiveness
    st.subheader("🏷️ Discount Effectiveness")
    if 'Discount' in df.columns and 'Units_Sold' in df.columns:
        discount_analysis = df.groupby(pd.cut(df['Discount'], bins=5))['Units_Sold'].mean().reset_index()
        discount_analysis['Discount_Range'] = discount_analysis['Discount'].astype(str)
        
        fig = px.bar(discount_analysis, x='Discount_Range', y='Units_Sold',
                    title="Average Sales by Discount Range")
        st.plotly_chart(fig, use_container_width=True)

def show_external_factors(conn, df):
    """Display external factors analysis"""
    st.header("🌤️ External Factors Impact")
    
    # Weather impact (if weather data exists)
    if 'Weather_Condition' in df.columns:
        st.subheader("🌦️ Weather Impact on Sales")
        weather_sales = df.groupby('Weather_Condition')['Units_Sold'].mean().reset_index()
        
        fig = px.bar(weather_sales, x='Weather_Condition', y='Units_Sold',
                    title="Average Sales by Weather Condition")
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    if 'Date' in df.columns:
        st.subheader("📅 Seasonal Analysis")
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        monthly_sales = df.groupby('Month')['Units_Sold'].sum().reset_index()
        
        fig = px.line(monthly_sales, x='Month', y='Units_Sold',
                     title="Monthly Sales Pattern")
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(conn, df):
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
