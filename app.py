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

# Initialize session state for data management
if 'data_modified' not in st.session_state:
    st.session_state.data_modified = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

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
        st.error("inventory_data.csv file not found. Please upload the file.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_current_data():
    """Get current data (either original or modified)"""
    if st.session_state.data_modified and st.session_state.current_data is not None:
        return st.session_state.current_data
    else:
        return load_data()

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
    if df is None or df.empty:
        return df
        
    filtered_df = df.copy()
    
    # Apply date filter
    if date_range and 'Date' in df.columns:
        if len(date_range) == 2:
            start_date, end_date = date_range
            try:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Date']).dt.date >= start_date) & 
                    (pd.to_datetime(filtered_df['Date']).dt.date <= end_date)
                ]
            except:
                pass  # Skip date filtering if conversion fails
    
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
    try:
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        existing_tables = run_sql_query(conn, tables_query)['name'].tolist()
    except:
        existing_tables = ['inventory_facts']
    
    # Check what columns exist in inventory_facts
    inventory_columns = get_table_columns(conn, 'inventory_facts')
    
    # Start building the query with proper table aliases
    joins = []
    where_conditions = []
    
    # Always use 'i' alias for inventory_facts
    from_clause = "inventory_facts i"
    
    # Handle region filter
    if regions:
        if 'stores' in existing_tables and 'Region' in get_table_columns(conn, 'stores'):
            # Join with stores table
            joins.append("LEFT JOIN stores s ON i.Store_ID = s.Store_ID")
            region_list = "', '".join(regions)
            where_conditions.append(f"s.Region IN ('{region_list}')")
        elif 'Region' in inventory_columns:
            # Use Region directly from inventory_facts
            region_list = "', '".join(regions)
            where_conditions.append(f"i.Region IN ('{region_list}')")
    
    # Handle category filter
    if categories:
        if 'products' in existing_tables and 'Category' in get_table_columns(conn, 'products'):
            # Join with products table
            joins.append("LEFT JOIN products p ON i.Product_ID = p.Product_ID")
            category_list = "', '".join(categories)
            where_conditions.append(f"p.Category IN ('{category_list}')")
        elif 'Category' in inventory_columns:
            # Use Category directly from inventory_facts
            category_list = "', '".join(categories)
            where_conditions.append(f"i.Category IN ('{category_list}')")
    
    # Handle date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        where_conditions.append(f"i.Date BETWEEN '{start_date}' AND '{end_date}'")
    
    # Build FROM clause with joins
    if joins:
        from_clause += " " + " ".join(joins)
    
    # Build WHERE clause
    where_clause = ""
    if where_conditions:
        where_clause = " WHERE " + " ".join(where_conditions)
    
    # Fix column references in SELECT clause to use table aliases
    fixed_select = base_select
    
    # Replace common column references with proper aliases
    column_mappings = {
        'Product_ID': 'i.Product_ID',
        'Store_ID': 'i.Store_ID', 
        'Units_Sold': 'i.Units_Sold',
        'Price': 'i.Price',
        'Inventory_Level': 'i.Inventory_Level',
        'Demand_Forecast': 'i.Demand_Forecast',
        'Date': 'i.Date'
    }
    
    # Only replace if not already aliased
    for col, aliased_col in column_mappings.items():
        if col in fixed_select and f'i.{col}' not in fixed_select and f's.{col}' not in fixed_select and f'p.{col}' not in fixed_select:
            # Use word boundaries to avoid partial replacements
            import re
            pattern = r'\b' + re.escape(col) + r'\b'
            fixed_select = re.sub(pattern, aliased_col, fixed_select)
    
    # Fix GROUP BY and ORDER BY clauses
    if 'GROUP BY Product_ID' in fixed_select:
        fixed_select = fixed_select.replace('GROUP BY Product_ID', 'GROUP BY i.Product_ID')
    if 'ORDER BY' in fixed_select and 'i.' not in fixed_select.split('ORDER BY')[1]:
        order_part = fixed_select.split('ORDER BY')[1]
        for col in column_mappings:
            if col in order_part and f'i.{col}' not in order_part:
                order_part = order_part.replace(col, f'i.{col}')
        fixed_select = fixed_select.split('ORDER BY')[0] + 'ORDER BY' + order_part
    
    final_query = f"{fixed_select} FROM {from_clause}{where_clause}"
    return final_query

def show_data_management(conn, df):
    """Display data management interface"""
    st.header("🔧 Data Management")
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    tab1, tab2, tab3 = st.tabs(["➕ Add Data", "🗑️ Delete Data", "📋 View Current Data"])
    
    with tab1:
        st.subheader("Add New Data Point")
        
        # Create form for adding new data
        with st.form("add_data_form"):
            cols = st.columns(3)
            new_data = {}
            
            # Get all columns from the dataframe
            for i, column in enumerate(df.columns):
                col_idx = i % 3
                with cols[col_idx]:
                    if df[column].dtype in ['object', 'string']:
                        # For string columns, provide text input with existing values as options
                        unique_values = df[column].dropna().unique().tolist()
                        if len(unique_values) <= 20:  # If not too many unique values, show selectbox
                            selected_value = st.selectbox(f"{column}", 
                                                        options=[""] + unique_values, 
                                                        key=f"add_{column}")
                            if selected_value == "":
                                new_data[column] = st.text_input(f"Or enter new {column}", key=f"add_text_{column}")
                            else:
                                new_data[column] = selected_value
                        else:
                            new_data[column] = st.text_input(f"{column}", key=f"add_{column}")
                    elif df[column].dtype in ['int64', 'float64']:
                        # For numeric columns
                        min_val = float(df[column].min()) if not df[column].empty else 0.0
                        max_val = float(df[column].max()) if not df[column].empty else 100.0
                        new_data[column] = st.number_input(f"{column}", 
                                                         min_value=min_val, 
                                                         max_value=max_val*2,
                                                         value=min_val,
                                                         key=f"add_{column}")
                    elif 'date' in column.lower():
                        # For date columns
                        new_data[column] = st.date_input(f"{column}", key=f"add_{column}")
                    else:
                        new_data[column] = st.text_input(f"{column}", key=f"add_{column}")
            
            submitted = st.form_submit_button("Add Data Point")
            
            if submitted:
                # Create new row
                new_row = pd.DataFrame([new_data])
                
                # Get current data
                current_df = get_current_data()
                if current_df is not None:
                    # Append new row
                    updated_df = pd.concat([current_df, new_row], ignore_index=True)
                    
                    # Update session state
                    st.session_state.current_data = updated_df
                    st.session_state.data_modified = True
                    
                    st.success("✅ Data point added successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Delete Data Points")
        
        current_df = get_current_data()
        if current_df is not None and not current_df.empty:
            # Show current data with selection
            st.write("Select rows to delete:")
            
            # Create a selection interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display data with index
                display_df = current_df.copy()
                display_df.insert(0, 'Row_Index', range(len(display_df)))
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                st.write("Delete Options:")
                
                # Option 1: Delete by row index
                max_index = len(current_df) - 1
                row_to_delete = st.number_input("Row Index to Delete", 
                                              min_value=0, 
                                              max_value=max_index, 
                                              value=0)
                
                if st.button("Delete Selected Row"):
                    if 0 <= row_to_delete <= max_index:
                        updated_df = current_df.drop(index=row_to_delete).reset_index(drop=True)
                        st.session_state.current_data = updated_df
                        st.session_state.data_modified = True
                        st.success(f"✅ Row {row_to_delete} deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid row index")
                
                st.markdown("---")
                
                # Option 2: Delete by condition
                st.write("Delete by Condition:")
                if len(current_df.columns) > 0:
                    delete_column = st.selectbox("Select Column", current_df.columns)
                    
                    if current_df[delete_column].dtype in ['object', 'string']:
                        unique_vals = current_df[delete_column].dropna().unique()
                        delete_value = st.selectbox("Select Value to Delete", unique_vals)
                    else:
                        delete_value = st.text_input("Enter Value to Delete")
                    
                    if st.button("Delete Matching Rows"):
                        try:
                            if current_df[delete_column].dtype in ['int64', 'float64']:
                                delete_value = float(delete_value)
                            
                            rows_to_delete = current_df[current_df[delete_column] == delete_value]
                            
                            if not rows_to_delete.empty:
                                updated_df = current_df[current_df[delete_column] != delete_value].reset_index(drop=True)
                                st.session_state.current_data = updated_df
                                st.session_state.data_modified = True
                                st.success(f"✅ Deleted {len(rows_to_delete)} rows where {delete_column} = {delete_value}")
                                st.rerun()
                            else:
                                st.warning("No matching rows found")
                        except Exception as e:
                            st.error(f"Error deleting rows: {str(e)}")
        else:
            st.info("No data available to delete")
    
    with tab3:
        st.subheader("Current Dataset")
        
        current_df = get_current_data()
        if current_df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(current_df))
            with col2:
                st.metric("Total Columns", len(current_df.columns))
            with col3:
                if st.session_state.data_modified:
                    st.metric("Status", "Modified", delta="Changes Made")
                else:
                    st.metric("Status", "Original", delta="No Changes")
            
            # Display current data
            st.dataframe(current_df, use_container_width=True)
            
            # Reset to original data button
            if st.session_state.data_modified:
                if st.button("🔄 Reset to Original Data"):
                    st.session_state.data_modified = False
                    st.session_state.current_data = None
                    st.success("Data reset to original!")
                    st.rerun()
        else:
            st.warning("No data available")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏪 Inventory Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = get_current_data()
        
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
         "💲 Price Optimization", "🌤️ External Factors", "🔍 Advanced Analytics", "🔧 Data Management"]
    )
    
    # Common filters
    st.sidebar.subheader("🔧 Filters")
    
    # Date range filter
    date_range = None
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(df['Date'].min().date(), df['Date'].max().date()),
                min_value=df['Date'].min().date(),
                max_value=df['Date'].max().date()
            )
        except:
            st.sidebar.warning("Date column format not recognized")
    
    # Region filter - only show if Region column exists
    regions = None
    if 'Region' in df.columns:
        unique_regions = df['Region'].dropna().unique()
        regions = st.sidebar.multiselect(
            "Select Regions",
            options=unique_regions,
            default=unique_regions
        )
    
    # Category filter - only show if Category column exists
    categories = None
    if 'Category' in df.columns:
        unique_categories = df['Category'].dropna().unique()
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=unique_categories,
            default=unique_categories
        )
    
    # Apply filters to dataframe
    filtered_df = apply_filters(df, date_range, regions, categories)
    
    # Show filter summary
    if len(filtered_df) != len(df):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
        if len(df) > 0:
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
    elif page == "🔧 Data Management":
        show_data_management(conn, df)

def show_overview(conn, df, date_range=None, regions=None, categories=None):
    """Display overview dashboard"""
    st.header("📊 Inventory Overview")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        query = build_filtered_query(conn, "SELECT COUNT(DISTINCT i.Product_ID) as count", date_range, regions, categories)
        total_skus = run_sql_query(conn, query)
        if not total_skus.empty:
            st.metric("Total SKUs", f"{total_skus.iloc[0]['count']:,}")
        else:
            # Fallback to pandas
            st.metric("Total SKUs", f"{df['Product_ID'].nunique():,}" if 'Product_ID' in df.columns else "N/A")
    
    with col2:
        query = build_filtered_query(conn, "SELECT COUNT(DISTINCT i.Store_ID) as count", date_range, regions, categories)
        total_stores = run_sql_query(conn, query)
        if not total_stores.empty:
            st.metric("Total Stores", f"{total_stores.iloc[0]['count']:,}")
        else:
            # Fallback to pandas
            st.metric("Total Stores", f"{df['Store_ID'].nunique():,}" if 'Store_ID' in df.columns else "N/A")
    
    with col3:
        if 'Units_Sold' in df.columns:
            total_sales = df['Units_Sold'].sum()
            st.metric("Total Units Sold", f"{total_sales:,}")
        else:
            st.metric("Total Units Sold", "N/A")
    
    with col4:
        if 'Price' in df.columns and 'Units_Sold' in df.columns:
            total_revenue = (df['Price'] * df['Units_Sold']).sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        else:
            st.metric("Total Revenue", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        if 'Category' in df.columns and 'Units_Sold' in df.columns:
            category_sales = df.groupby('Category')['Units_Sold'].sum().reset_index()
            if not category_sales.empty:
                fig = px.pie(category_sales, values='Units_Sold', names='Category', 
                            title="Sales Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or Units_Sold data not available")
    
    with col2:
        st.subheader("Sales by Region")
        if 'Region' in df.columns and 'Units_Sold' in df.columns:
            region_sales = df.groupby('Region')['Units_Sold'].sum().reset_index()
            if not region_sales.empty:
                fig = px.bar(region_sales, x='Region', y='Units_Sold',
                            title="Sales by Region")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region or Units_Sold data not available")
    
    # Inventory alerts
    st.subheader("🚨 Inventory Alerts")
    
    if 'Inventory_Level' in df.columns and 'Demand_Forecast' in df.columns:
        # Low stock alerts - using pandas for simplicity
        low_stock_df = df[df['Inventory_Level'] < df['Demand_Forecast']]
        
        if not low_stock_df.empty:
            st.warning(f"⚠️ {len(low_stock_df)} products are understocked")
            with st.expander("View Understocked Items"):
                display_columns = ['Product_ID', 'Store_ID', 'Inventory_Level', 'Demand_Forecast']
                available_columns = [col for col in display_columns if col in low_stock_df.columns]
                st.dataframe(low_stock_df[available_columns].head(10))
        else:
            st.success("✅ No understocked items detected")
    else:
        st.info("Inventory level or demand forecast data not available for alerts")

def show_sales_analysis(conn, df, date_range=None, regions=None, categories=None):
    """Display sales analysis dashboard"""
    st.header("💰 Sales Performance Analysis")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Top performing products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Products by Sales")
        if 'Units_Sold' in df.columns and 'Product_ID' in df.columns:
            query = build_filtered_query(conn, "SELECT i.Product_ID, SUM(i.Units_Sold) as Total_Sales", date_range, regions, categories)
            query += " GROUP BY i.Product_ID ORDER BY Total_Sales DESC LIMIT 10"
            top_products = run_sql_query(conn, query)
            
            if not top_products.empty:
                fig = px.bar(top_products, x='Product_ID', y='Total_Sales',
                            title="Top 10 Products by Units Sold")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to pandas
                top_products_pd = df.groupby('Product_ID')['Units_Sold'].sum().reset_index()
                top_products_pd = top_products_pd.sort_values('Units_Sold', ascending=False).head(10)
                if not top_products_pd.empty:
                    fig = px.bar(top_products_pd, x='Product_ID', y='Units_Sold',
                                title="Top 10 Products by Units Sold")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product_ID or Units_Sold data not available")
    
    with col2:
        st.subheader("💸 Revenue by Product")
        if 'Price' in df.columns and 'Units_Sold' in df.columns and 'Product_ID' in df.columns:
            query = build_filtered_query(conn, "SELECT i.Product_ID, SUM(i.Units_Sold * i.Price) as Revenue", date_range, regions, categories)
            query += " GROUP BY i.Product_ID ORDER BY Revenue DESC LIMIT 10"
            revenue_data = run_sql_query(conn, query)
            
            if not revenue_data.empty:
                fig = px.bar(revenue_data, x='Product_ID', y='Revenue',
                            title="Top 10 Products by Revenue")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to pandas
                df['Revenue'] = df['Price'] * df['Units_Sold']
                revenue_pd = df.groupby('Product_ID')['Revenue'].sum().reset_index()
                revenue_pd = revenue_pd.sort_values('Revenue', ascending=False).head(10)
                if not revenue_pd.empty:
                    fig = px.bar(revenue_pd, x='Product_ID', y='Revenue',
                                title="Top 10 Products by Revenue")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price, Units_Sold, or Product_ID data not available")
    
    # Sales trends over time
    st.subheader("📈 Sales Trends Over Time")
    if 'Date' in df.columns and 'Units_Sold' in df.columns:
        try:
            daily_sales = df.groupby('Date')['Units_Sold'].sum().reset_index()
            if not daily_sales.empty:
                fig = px.line(daily_sales, x='Date', y='Units_Sold',
                             title="Daily Sales Trend")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating sales trend chart: {str(e)}")
    else:
        st.info("Date or Units_Sold data not available")

def show_inventory_management(conn, df, date_range=None, regions=None, categories=None):
    """Display inventory management dashboard"""
    st.header("📦 Inventory Management")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Use pandas for inventory analysis to avoid SQL complexity
    if 'Inventory_Level' in df.columns and 'Units_Sold' in df.columns and 'Product_ID' in df.columns:
        st.subheader("🔄 Inventory Turnover Analysis")
        
        try:
            turnover_df = df.groupby('Product_ID').agg({
                'Units_Sold': 'sum',
                'Inventory_Level': 'mean'
            }).reset_index()
            
            # Avoid division by zero
            turnover_df['Turnover_Ratio'] = turnover_df['Units_Sold'] / turnover_df['Inventory_Level'].replace(0, np.nan)
            turnover_df = turnover_df.dropna().sort_values('Turnover_Ratio', ascending=False).head(15)
            
            if not turnover_df.empty:
                fig = px.bar(turnover_df, x='Product_ID', y='Turnover_Ratio',
                            title="Product Turnover Ratios")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid turnover data available")
        except Exception as e:
            st.error(f"Error calculating turnover ratios: {str(e)}")
    else:
        st.info("Required columns (Inventory_Level, Units_Sold, Product_ID) not available")

def show_price_optimization(conn, df, date_range=None, regions=None, categories=None):
    """Display price optimization dashboard"""
    st.header("💲 Price Optimization Analysis")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Price elasticity analysis
    st.subheader("📊 Price vs Sales Analysis")
    
    if 'Price' in df.columns and 'Units_Sold' in df.columns and 'Product_ID' in df.columns:
        try:
            # Sweet spot analysis using pandas
            sweet_spot_data = df.groupby(['Product_ID', 'Price'])['Units_Sold'].mean().reset_index()
            sweet_spot_data.columns = ['Product_ID', 'Price', 'Avg_Sales']
            
            if not sweet_spot_data.empty:
                # Select a product for detailed analysis
                products = sweet_spot_data['Product_ID'].unique()
                if len(products) > 0:
                    selected_product = st.selectbox("Select Product for Price Analysis", products)
                    
                    product_data = sweet_spot_data[sweet_spot_data['Product_ID'] == selected_product]
                    
                    if not product_data.empty:
                        fig = px.scatter(product_data, x='Price', y='Avg_Sales',
                                       title=f"Price vs Sales for {selected_product}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No data available for product {selected_product}")
                else:
                    st.info("No products available for analysis")
            else:
                st.info("No price-sales data available")
        except Exception as e:
            st.error(f"Error in price optimization analysis: {str(e)}")
    else:
        st.info("Required columns (Price, Units_Sold, Product_ID) not available")

def show_external_factors(conn, df, date_range=None, regions=None, categories=None):
    """Display external factors analysis"""
    st.header("🌤️ External Factors Impact")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Weather impact (if weather data exists)
    if 'Weather_Condition' in df.columns and 'Units_Sold' in df.columns:
        st.subheader("🌦️ Weather Impact on Sales")
        try:
            weather_sales = df.groupby('Weather_Condition')['Units_Sold'].mean().reset_index()
            
            if not weather_sales.empty:
                fig = px.bar(weather_sales, x='Weather_Condition', y='Units_Sold',
                            title="Average Sales by Weather Condition")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing weather impact: {str(e)}")
    else:
        st.info("Weather_Condition or Units_Sold data not available")
    
    # Seasonal analysis
    if 'Date' in df.columns and 'Units_Sold' in df.columns:
        st.subheader("📅 Seasonal Analysis")
        try:
            df_temp = df.copy()
            df_temp['Month'] = pd.to_datetime(df_temp['Date']).dt.month
            monthly_sales = df_temp.groupby('Month')['Units_Sold'].sum().reset_index()
            
            if not monthly_sales.empty:
                fig = px.line(monthly_sales, x='Month', y='Units_Sold',
                             title="Monthly Sales Pattern")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in seasonal analysis: {str(e)}")
    else:
        st.info("Date or Units_Sold data not available")
        

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
