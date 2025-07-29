import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SuperStore Analytics Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load and preprocess the SuperStore dataset"""
    try:
        # Replace 'SuperStoreUS.csv' with your actual file path
        df = pd.read_csv('../datasets/SuperStoreUS.csv')
        
        # Data preprocessing
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        # Create additional calculated fields
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Month_Year'] = df['Order Date'].dt.to_period('M')
        df['Delivery_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
        df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
        df['Shipping_Cost_Percentage'] = (df['Shipping Cost'] / df['Sales']) * 100
        
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'SuperStoreUS.csv' is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-info"><h2>🏪 SuperStore Analytics</h2><p>Navigate through different sections of the dashboard</p></div>', unsafe_allow_html=True)

# Navigation menu
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home & Overview", "📊 Analytics Dashboard", "📈 Performance Insights", "💡 Business Intelligence", "📋 Detailed Reports"]
)

# Sidebar filters (available on all pages)
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Order Date'].min(), df['Order Date'].max()),
        min_value=df['Order Date'].min(),
        max_value=df['Order Date'].max()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Customer segment filter
    segments = st.sidebar.multiselect(
        "Select Customer Segments",
        options=df['Customer Segment'].unique(),
        default=df['Customer Segment'].unique()
    )
    
    # Product category filter
    categories = st.sidebar.multiselect(
        "Select Product Categories",
        options=df['Product Category'].unique(),
        default=df['Product Category'].unique()
    )
    
    # Apply filters
    mask = (
        (df['Order Date'] >= pd.to_datetime(date_range[0])) &
        (df['Order Date'] <= pd.to_datetime(date_range[1])) &
        (df['Region'].isin(regions)) &
        (df['Customer Segment'].isin(segments)) &
        (df['Product Category'].isin(categories))
    )
    filtered_df = df[mask]
    
    # Show filter info
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} of {len(df):,}")

# Main content area
if df is None:
    st.error("Unable to load data. Please check your data file.")
    st.stop()

# ============================================================================
# PAGE 1: HOME & OVERVIEW
# ============================================================================
if page == "🏠 Home & Overview":
    st.markdown('<h1 class="main-header">🏪 SuperStore Business Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sales = filtered_df['Sales'].sum()
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        total_profit = filtered_df['Profit'].sum()
        profit_growth = ((total_profit / df['Profit'].sum()) - 1) * 100 if len(df) > len(filtered_df) else 0
        st.metric("📈 Total Profit", f"${total_profit:,.0f}", f"{profit_growth:+.1f}%")
    
    with col3:
        total_orders = filtered_df['Order ID'].nunique()
        st.metric("🛒 Total Orders", f"{total_orders:,}")
    
    with col4:
        avg_order_value = filtered_df['Sales'].mean()
        st.metric("💳 Avg Order Value", f"${avg_order_value:.0f}")
    
    with col5:
        profit_margin = (total_profit / total_sales) * 100
        st.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
    
    st.markdown("---")
    
    # Dataset overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Dataset Overview")
        overview_data = {
            "Metric": [
                "Total Records", "Date Range", "Unique Customers", "Unique Products",
                "Regions", "Customer Segments", "Product Categories", "Ship Modes"
            ],
            "Value": [
                f"{len(filtered_df):,}",
                f"{filtered_df['Order Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Order Date'].max().strftime('%Y-%m-%d')}",
                f"{filtered_df['Customer ID'].nunique():,}",
                f"{filtered_df['Product Name'].nunique():,}",
                f"{filtered_df['Region'].nunique()}",
                f"{filtered_df['Customer Segment'].nunique()}",
                f"{filtered_df['Product Category'].nunique()}",
                f"{filtered_df['Ship Mode'].nunique()}"
            ]
        }
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Quick Insights")
        
        # Top region by sales
        top_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
        top_region_sales = filtered_df.groupby('Region')['Sales'].sum().max()
        
        # Top product category
        top_category = filtered_df.groupby('Product Category')['Sales'].sum().idxmax()
        
        # Best customer segment
        top_segment = filtered_df.groupby('Customer Segment')['Profit'].sum().idxmax()
        
        st.info(f"🏆 **Best Region:** {top_region} (${top_region_sales:,.0f})")
        st.info(f"📦 **Top Category:** {top_category}")
        st.info(f"👥 **Best Segment:** {top_segment}")
        
        # Health indicator
        if profit_margin >= 15:
            st.success("✅ **Business Health:** Excellent")
        elif profit_margin >= 10:
            st.warning("⚠️ **Business Health:** Good")
        else:
            st.error("🚨 **Business Health:** Needs Attention")
    
    # Sample data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Data quality information
    st.subheader("🔧 Data Quality")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_data = filtered_df.isnull().sum().sum()
        st.info(f"Missing Values: {missing_data}")
    
    with col2:
        duplicate_orders = filtered_df['Order ID'].duplicated().sum()
        st.info(f"Duplicate Orders: {duplicate_orders}")
    
    with col3:
        negative_profits = len(filtered_df[filtered_df['Profit'] < 0])
        st.info(f"Loss-Making Orders: {negative_profits}")

# ============================================================================
# PAGE 2: ANALYTICS DASHBOARD
# ============================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Analysis", "💰 Profitability", "🌍 Geographic", "👥 Customer"])
    
    with tab1:
        st.subheader("📈 Sales Performance Analysis")
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly sales trend
            monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum()
            fig = px.line(
                x=monthly_sales.index.astype(str), 
                y=monthly_sales.values,
                title="Monthly Sales Trend",
                labels={'x': 'Month', 'y': 'Sales ($)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sales by product category
            cat_sales = filtered_df.groupby('Product Category')['Sales'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=cat_sales.index, 
                y=cat_sales.values,
                title="Sales by Product Category",
                labels={'x': 'Product Category', 'y': 'Sales ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sales by region and segment
        col1, col2 = st.columns(2)
        
        with col1:
            region_sales = filtered_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            fig = px.pie(
                values=region_sales.values, 
                names=region_sales.index,
                title="Sales Distribution by Region"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_sales = filtered_df.groupby('Customer Segment')['Sales'].sum()
            fig = px.bar(
                x=segment_sales.index, 
                y=segment_sales.values,
                title="Sales by Customer Segment",
                labels={'x': 'Customer Segment', 'y': 'Sales ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Profitability Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit by category
            cat_profit = filtered_df.groupby('Product Category')['Profit'].sum()
            fig = px.bar(
                x=cat_profit.index, 
                y=cat_profit.values,
                title="Profit by Product Category",
                labels={'x': 'Product Category', 'y': 'Profit ($)'},
                color=cat_profit.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit margins by category
            cat_margins = filtered_df.groupby('Product Category').apply(
                lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100
            )
            fig = px.bar(
                x=cat_margins.index, 
                y=cat_margins.values,
                title="Profit Margins by Category",
                labels={'x': 'Product Category', 'y': 'Profit Margin (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Profit vs Sales scatter
        fig = px.scatter(
            filtered_df, 
            x='Sales', 
            y='Profit',
            color='Product Category',
            title="Profit vs Sales Relationship",
            hover_data=['Customer Segment', 'Region']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🌍 Geographic Analysis")
        
        # Geographic performance
        geo_data = filtered_df.groupby(['Region', 'State or Province']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional performance
            region_perf = filtered_df.groupby('Region').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Sales', x=region_perf.index, y=region_perf['Sales']))
            fig.add_trace(go.Bar(name='Profit', x=region_perf.index, y=region_perf['Profit']))
            fig.update_layout(title="Regional Sales vs Profit", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top states by sales
            state_sales = filtered_df.groupby('State or Province')['Sales'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=state_sales.values, 
                y=state_sales.index,
                orientation='h',
                title="Top 10 States by Sales",
                labels={'x': 'Sales ($)', 'y': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("👥 Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segment analysis
            segment_analysis = filtered_df.groupby('Customer Segment').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Customer ID': 'nunique',
                'Order ID': 'nunique'
            })
            segment_analysis['Avg_Order_Value'] = segment_analysis['Sales'] / segment_analysis['Order ID']
            segment_analysis['Profit_per_Customer'] = segment_analysis['Profit'] / segment_analysis['Customer ID']
            
            fig = px.scatter(
                x=segment_analysis['Avg_Order_Value'],
                y=segment_analysis['Profit_per_Customer'],
                size=segment_analysis['Sales'],
                color=segment_analysis.index,
                title="Customer Segment Performance",
                labels={'x': 'Avg Order Value ($)', 'y': 'Profit per Customer ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top customers by sales
            top_customers = filtered_df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_customers.values,
                y=top_customers.index,
                orientation='h',
                title="Top 10 Customers by Sales",
                labels={'x': 'Sales ($)', 'y': 'Customer'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Placeholder for remaining pages - we'll build these in the next steps
else:
    st.markdown(f'<h1 class="main-header">{page}</h1>', unsafe_allow_html=True)
    st.info("🚧 This page is under construction. We'll build it in the next steps!")
    
    if page == "📈 Performance Insights":
        st.markdown("""
        **Coming up in this section:**
        - Detailed performance metrics
        - Trend analysis and forecasting
        - Comparative analysis
        - Key performance indicators
        """)
    
    elif page == "💡 Business Intelligence":
        st.markdown("""
        **Coming up in this section:**
        - Strategic recommendations
        - Business insights
        - Action items
        - Growth opportunities
        """)
    
    elif page == "📋 Detailed Reports":
        st.markdown("""
        **Coming up in this section:**
        - Comprehensive data tables
        - Export functionality
        - Custom reports
        - Data downloads
        """)

# Footer
st.markdown("---")
st.markdown("*SuperStore Analytics Dashboard - Built with Streamlit*")