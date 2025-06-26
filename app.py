# Add this function after show_advanced_analytics() and before main()

def show_data_management():
    """Display data management dashboard"""
    st.header("🔧 Data Management")
    
    # Initialize session state for data if not exists
    if 'df_data' not in st.session_state:
        st.session_state.df_data = load_data()
    
    df = st.session_state.df_data
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    tab1, tab2 = st.tabs(["➕ Add Data", "🗑️ Delete Data"])
    
    with tab1:
        st.subheader("Add New Record")
        
        # Get available columns
        columns = df.columns.tolist()
        
        # Create form for adding new data
        with st.form("add_data_form"):
            new_data = {}
            
            # Create input fields for each column
            for col in columns:
                if 'date' in col.lower() or 'Date' in col:
                    # Use date input for date columns
                    new_data[col] = st.date_input(f"{col}")
                elif df[col].dtype in ['int64', 'float64']:
                    new_data[col] = st.number_input(f"{col}", value=0.0)
                else:
                    # Get unique values for dropdown
                    unique_vals = df[col].dropna().unique().tolist()
                    if len(unique_vals) < 50:  # Use selectbox for limited options
                        new_data[col] = st.selectbox(f"{col}", [""] + unique_vals)
                    else:
                        new_data[col] = st.text_input(f"{col}")
            
            submitted = st.form_submit_button("Add Record")
            
            if submitted:
                try:
                    # Convert date columns to string format for SQLite compatibility
                    for col in new_data:
                        if 'date' in col.lower() or 'Date' in col:
                            if hasattr(new_data[col], 'strftime'):
                                new_data[col] = new_data[col].strftime('%Y-%m-%d')
                    
                    # Add to session state dataframe
                    new_row = pd.DataFrame([new_data])
                    st.session_state.df_data = pd.concat([st.session_state.df_data, new_row], ignore_index=True)
                    st.success("✅ Record added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding record: {str(e)}")
    
    with tab2:
        st.subheader("Delete Records")
        
        # Show current data with selection
        st.write("Select records to delete:")
        
        # Create a subset for display (first 100 rows)
        display_df = df.head(100).copy()
        display_df.insert(0, 'Select', False)
        
        # Use data editor for selection
        edited_df = st.data_editor(
            display_df,
            column_config={"Select": st.column_config.CheckboxColumn("Select")},
            disabled=[col for col in display_df.columns if col != 'Select'],
            hide_index=True,
            use_container_width=True
        )
        
        # Delete selected records
        if st.button("🗑️ Delete Selected"):
            selected_rows = edited_df[edited_df['Select'] == True]
            
            if len(selected_rows) > 0:
                try:
                    # Get indices to drop
                    indices_to_drop = selected_rows.index.tolist()
                    
                    # Drop from session state dataframe
                    st.session_state.df_data = st.session_state.df_data.drop(indices_to_drop).reset_index(drop=True)
                    
                    st.success(f"✅ Deleted {len(selected_rows)} records!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error deleting records: {str(e)}")
            else:
                st.warning("No records selected for deletion")

# Modify the load_data function to use session state:
# Replace your existing load_data function with this:

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
    """Get current data from session state or load fresh"""
    if 'df_data' in st.session_state:
        return st.session_state.df_data
    else:
        st.session_state.df_data = load_data()
        return st.session_state.df_data

# Update main() function - replace the data loading section with:
    # Load data
    with st.spinner("Loading data..."):
        df = get_current_data()

# And change the data management call to:
    elif page == "🔧 Data Management":
        show_data_management()
