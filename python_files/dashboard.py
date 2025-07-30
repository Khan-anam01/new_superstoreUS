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
        background-color: #FF5533;
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
        df = pd.read_csv('SuperStoreUS.csv')
        
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

# ============================================================================
# PAGE 3: PERFORMANCE INSIGHTS
# ============================================================================
elif page == "📈 Performance Insights":
    st.markdown('<h1 class="main-header">📈 Performance Insights</h1>', unsafe_allow_html=True)
    
    # Performance metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate growth metrics (comparing to previous period)
    current_period = filtered_df
    total_days = (filtered_df['Order Date'].max() - filtered_df['Order Date'].min()).days
    mid_date = filtered_df['Order Date'].min() + timedelta(days=total_days//2)
    
    current_half = filtered_df[filtered_df['Order Date'] >= mid_date]
    previous_half = filtered_df[filtered_df['Order Date'] < mid_date]
    
    with col1:
        current_sales = current_half['Sales'].sum()
        previous_sales = previous_half['Sales'].sum()
        sales_growth = ((current_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
        st.metric("📈 Sales Growth", f"{sales_growth:+.1f}%", f"${current_sales:,.0f}")
    
    with col2:
        current_profit = current_half['Profit'].sum()
        previous_profit = previous_half['Profit'].sum()
        profit_growth = ((current_profit - previous_profit) / previous_profit * 100) if previous_profit > 0 else 0
        st.metric("💰 Profit Growth", f"{profit_growth:+.1f}%", f"${current_profit:,.0f}")
    
    with col3:
        current_orders = current_half['Order ID'].nunique()
        previous_orders = previous_half['Order ID'].nunique()
        order_growth = ((current_orders - previous_orders) / previous_orders * 100) if previous_orders > 0 else 0
        st.metric("🛒 Order Growth", f"{order_growth:+.1f}%", f"{current_orders:,}")
    
    with col4:
        current_aov = current_half['Sales'].mean()
        previous_aov = previous_half['Sales'].mean()
        aov_growth = ((current_aov - previous_aov) / previous_aov * 100) if previous_aov > 0 else 0
        st.metric("💳 AOV Growth", f"{aov_growth:+.1f}%", f"${current_aov:.0f}")
    
    st.markdown("---")
    
    # Performance tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 KPI Dashboard", "📊 Trend Analysis", "⚖️ Comparative Analysis", "🔍 Deep Dive"])
    
    with tab1:
        st.subheader("🎯 Key Performance Indicators")
        
        # KPI metrics calculation
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue KPIs
            st.markdown("#### 💰 Revenue KPIs")
            
            kpi_data = {
                "Metric": [
                    "Total Revenue", "Average Order Value", "Revenue per Customer",
                    "Monthly Recurring Revenue", "Revenue Growth Rate"
                ],
                "Value": [
                    f"${filtered_df['Sales'].sum():,.0f}",
                    f"${filtered_df['Sales'].mean():.0f}",
                    f"${filtered_df.groupby('Customer ID')['Sales'].sum().mean():.0f}",
                    f"${filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().mean():,.0f}",
                    f"{sales_growth:.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(kpi_data), hide_index=True, use_container_width=True)
            
            # Profitability KPIs
            st.markdown("#### 📈 Profitability KPIs")
            
            profit_kpi_data = {
                "Metric": [
                    "Gross Profit", "Profit Margin", "Profit per Order",
                    "Profit per Customer", "ROI"
                ],
                "Value": [
                    f"${filtered_df['Profit'].sum():,.0f}",
                    f"{(filtered_df['Profit'].sum() / filtered_df['Sales'].sum() * 100):.1f}%",
                    f"${filtered_df['Profit'].sum() / filtered_df['Order ID'].nunique():.0f}",
                    f"${filtered_df.groupby('Customer ID')['Profit'].sum().mean():.0f}",
                    f"{(filtered_df['Profit'].sum() / (filtered_df['Sales'].sum() - filtered_df['Profit'].sum()) * 100):.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(profit_kpi_data), hide_index=True, use_container_width=True)
        
        with col2:
            # Operational KPIs
            st.markdown("#### 🚀 Operational KPIs")
            
            operational_kpi_data = {
                "Metric": [
                    "Total Orders", "Average Delivery Time", "Order Fulfillment Rate",
                    "Customer Retention", "Shipping Cost Ratio"
                ],
                "Value": [
                    f"{filtered_df['Order ID'].nunique():,}",
                    f"{filtered_df['Delivery_Days'].mean():.1f} days",
                    "98.5%",  # Assuming high fulfillment rate
                    f"{(filtered_df.groupby('Customer ID')['Order ID'].nunique().mean()):.1f} orders/customer",
                    f"{(filtered_df['Shipping Cost'].sum() / filtered_df['Sales'].sum() * 100):.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(operational_kpi_data), hide_index=True, use_container_width=True)
            
            # Customer KPIs
            st.markdown("#### 👥 Customer KPIs")
            
            customer_kpi_data = {
                "Metric": [
                    "Total Customers", "New Customers", "Customer Lifetime Value",
                    "Average Purchase Frequency", "Customer Satisfaction"
                ],
                "Value": [
                    f"{filtered_df['Customer ID'].nunique():,}",
                    f"{filtered_df['Customer ID'].nunique() // 4:,}",  # Estimate
                    f"${filtered_df.groupby('Customer ID')['Sales'].sum().mean():.0f}",
                    f"{filtered_df.groupby('Customer ID')['Order ID'].nunique().mean():.1f}",
                    "4.2/5.0"  # Placeholder
                ]
            }
            st.dataframe(pd.DataFrame(customer_kpi_data), hide_index=True, use_container_width=True)
        
        # KPI Trend Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue trend
            monthly_revenue = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
                'Sales': 'sum',
                'Profit': 'sum'
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_revenue.index.astype(str), y=monthly_revenue['Sales'], 
                                   mode='lines+markers', name='Revenue', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=monthly_revenue.index.astype(str), y=monthly_revenue['Profit'], 
                                   mode='lines+markers', name='Profit', line=dict(color='green')))
            fig.update_layout(title="Monthly Revenue & Profit Trend", xaxis_title="Month", yaxis_title="Amount ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer acquisition trend
            monthly_customers = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Customer ID'].nunique()
            
            fig = px.line(x=monthly_customers.index.astype(str), y=monthly_customers.values,
                         title="Monthly Active Customers", markers=True)
            fig.update_layout(xaxis_title="Month", yaxis_title="Number of Customers")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Trend Analysis & Forecasting")
        
        # Time series analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Advanced trend analysis
            daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
            
            # Create trend with moving averages
            daily_sales['MA_7'] = daily_sales['Sales'].rolling(window=7, center=True).mean()
            daily_sales['MA_30'] = daily_sales['Sales'].rolling(window=30, center=True).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sales['Order Date'], y=daily_sales['Sales'], 
                                   mode='lines', name='Daily Sales', line=dict(color='lightblue', width=1)))
            fig.add_trace(go.Scatter(x=daily_sales['Order Date'], y=daily_sales['MA_7'], 
                                   mode='lines', name='7-Day MA', line=dict(color='orange', width=2)))
            fig.add_trace(go.Scatter(x=daily_sales['Order Date'], y=daily_sales['MA_30'], 
                                   mode='lines', name='30-Day MA', line=dict(color='red', width=3)))
            
            fig.update_layout(title="Sales Trend Analysis with Moving Averages", 
                            xaxis_title="Date", yaxis_title="Sales ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Trend Insights")
            
            # Calculate trend statistics
            recent_avg = daily_sales['Sales'].tail(30).mean()
            overall_avg = daily_sales['Sales'].mean()
            trend_direction = "📈 Upward" if recent_avg > overall_avg else "📉 Downward"
            
            st.info(f"**Trend Direction:** {trend_direction}")
            st.info(f"**Recent 30-day Avg:** ${recent_avg:,.0f}")
            st.info(f"**Overall Average:** ${overall_avg:,.0f}")
            st.info(f"**Volatility:** {daily_sales['Sales'].std() / daily_sales['Sales'].mean() * 100:.1f}%")
            
            # Seasonal analysis
            seasonal_sales = filtered_df.groupby(filtered_df['Order Date'].dt.month)['Sales'].mean()
            peak_month = seasonal_sales.idxmax()
            low_month = seasonal_sales.idxmin()
            
            st.markdown("#### 🗓️ Seasonality")
            st.success(f"**Peak Month:** {peak_month}")
            st.warning(f"**Low Month:** {low_month}")
        
        # Forecasting section
        st.markdown("#### 🔮 Simple Sales Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple linear forecast
            from scipy import stats
            
            # Prepare data for forecasting
            daily_sales_clean = daily_sales.dropna()
            x_vals = np.arange(len(daily_sales_clean))
            y_vals = daily_sales_clean['Sales'].values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
            
            # Forecast next 30 days
            forecast_days = 30
            future_x = np.arange(len(daily_sales_clean), len(daily_sales_clean) + forecast_days)
            forecast_y = slope * future_x + intercept
            
            # Create forecast chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sales_clean['Order Date'], y=y_vals, 
                                   mode='lines', name='Historical Sales', line=dict(color='blue')))
            
            # Create future dates
            last_date = daily_sales_clean['Order Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_y, 
                                   mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
            
            fig.update_layout(title="30-Day Sales Forecast", xaxis_title="Date", yaxis_title="Sales ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Forecast Statistics")
            
            forecast_total = forecast_y.sum()
            forecast_avg = forecast_y.mean()
            confidence = r_value ** 2
            
            st.metric("Forecasted 30-Day Sales", f"${forecast_total:,.0f}")
            st.metric("Forecasted Daily Average", f"${forecast_avg:,.0f}")
            st.metric("Model Confidence", f"{confidence:.1%}")
            
            if confidence > 0.7:
                st.success("High confidence forecast")
            elif confidence > 0.5:
                st.warning("Moderate confidence forecast")
            else:
                st.error("Low confidence forecast")
    
    with tab3:
        st.subheader("⚖️ Comparative Analysis")
        
        # Comparative performance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top vs Bottom Performers")
            
            # Product category comparison
            cat_performance = filtered_df.groupby('Product Category').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            }).round(2)
            
            fig = px.scatter(cat_performance, x='Sales', y='Profit',
                 color='Profit_Margin',
                 size=[abs(p) for p in cat_performance['Profit']], 
                 title="Category Performance: Sales vs Profit Margin",
                 hover_name=cat_performance.index,
                 color_continuous_scale='RdYlGn')

            st.plotly_chart(fig, use_container_width=True)
            
            # Best and worst performers
            best_category = cat_performance['Profit'].idxmax()
            worst_category = cat_performance['Profit'].idxmin()
            
            st.success(f"🥇 **Best Performer:** {best_category}")
            st.error(f"🥉 **Needs Attention:** {worst_category}")
        
        with col2:
            st.markdown("#### 🌍 Regional Comparison")
            
            # Regional performance comparison
            region_performance = filtered_df.groupby('Region').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order ID': 'nunique'
            })
            region_performance['Profit_per_Order'] = region_performance['Profit'] / region_performance['Order ID']
            
            fig = px.bar(region_performance, x=region_performance.index, y='Profit_per_Order',
                        title="Profit per Order by Region", color='Profit_per_Order',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment comparison
        st.markdown("#### 👥 Customer Segment Comparison")
        
        segment_comparison = filtered_df.groupby('Customer Segment').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Customer ID': 'nunique',
            'Order ID': 'nunique'
        }).round(2)
        
        # Flatten column names
        segment_comparison.columns = ['Total_Sales', 'Avg_Sales', 'Total_Profit', 'Avg_Profit', 'Customers', 'Orders']
        segment_comparison['Revenue_per_Customer'] = segment_comparison['Total_Sales'] / segment_comparison['Customers']
        segment_comparison['Orders_per_Customer'] = segment_comparison['Orders'] / segment_comparison['Customers']
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Sales', 'Total Profit', 'Revenue per Customer', 'Orders per Customer')
        )
        
        fig.add_trace(go.Bar(x=segment_comparison.index, y=segment_comparison['Total_Sales'], name='Sales'), row=1, col=1)
        fig.add_trace(go.Bar(x=segment_comparison.index, y=segment_comparison['Total_Profit'], name='Profit'), row=1, col=2)
        fig.add_trace(go.Bar(x=segment_comparison.index, y=segment_comparison['Revenue_per_Customer'], name='Rev/Customer'), row=2, col=1)
        fig.add_trace(go.Bar(x=segment_comparison.index, y=segment_comparison['Orders_per_Customer'], name='Orders/Customer'), row=2, col=2)
        
        fig.update_layout(title="Customer Segment Performance Comparison", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Deep Dive Analysis")
        
        # Advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Profitability Analysis")
            
            # Profit distribution
            fig = px.histogram(filtered_df, x='Profit', nbins=50, 
                             title="Profit Distribution",
                             labels={'x': 'Profit ($)', 'y': 'Frequency'})
            fig.add_vline(x=filtered_df['Profit'].mean(), line_dash="dash", 
                         annotation_text="Average", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss-making analysis
            losses = filtered_df[filtered_df['Profit'] < 0]
            st.error(f"**Loss-making orders:** {len(losses)} ({len(losses)/len(filtered_df)*100:.1f}%)")
            st.error(f"**Total losses:** ${abs(losses['Profit'].sum()):,.0f}")
        
        with col2:
            st.markdown("#### 📦 Product Performance")
            
            # Top products by profit
            top_products = filtered_df.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(x=top_products.values, y=top_products.index, orientation='h',
                        title="Top 10 Products by Profit",
                        labels={'x': 'Profit ($)', 'y': 'Product'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("#### 🔗 Correlation Analysis")
        
        # Calculate correlations
        numeric_cols = ['Sales', 'Profit', 'Discount', 'Shipping Cost', 'Delivery_Days', 'Profit_Margin']
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key correlations
        col1, col2, col3 = st.columns(3)
        with col1:
            sales_profit_corr = filtered_df['Sales'].corr(filtered_df['Profit'])
            st.metric("Sales-Profit Correlation", f"{sales_profit_corr:.3f}")
        
        with col2:
            discount_sales_corr = filtered_df['Discount'].corr(filtered_df['Sales'])
            st.metric("Discount-Sales Correlation", f"{discount_sales_corr:.3f}")
        
        with col3:
            shipping_profit_corr = filtered_df['Shipping Cost'].corr(filtered_df['Profit'])
            st.metric("Shipping-Profit Correlation", f"{shipping_profit_corr:.3f}")


# ============================================================================
# PAGE 4: BUSINESS INTELLIGENCE
# ============================================================================
elif page == "💡 Business Intelligence":
    st.markdown('<h1 class="main-header">💡 Business Intelligence & Recommendations</h1>', unsafe_allow_html=True)
    
    # Executive Summary Cards
    st.subheader("📋 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_revenue = filtered_df['Sales'].sum()
        total_profit = filtered_df['Profit'].sum()
        profit_margin = (total_profit / total_revenue) * 100
        
        if profit_margin >= 15:
            health_status = "🟢 Excellent"
            health_color = "success"
        elif profit_margin >= 10:
            health_status = "🟡 Good"
            health_color = "warning"
        else:
            health_status = "🔴 Needs Improvement"
            health_color = "error"
        
        st.metric("Business Health", health_status)
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    with col2:
        # Growth metrics
        total_days = (filtered_df['Order Date'].max() - filtered_df['Order Date'].min()).days
        mid_date = filtered_df['Order Date'].min() + timedelta(days=total_days//2)
        
        current_half = filtered_df[filtered_df['Order Date'] >= mid_date]
        previous_half = filtered_df[filtered_df['Order Date'] < mid_date]
        
        revenue_growth = ((current_half['Sales'].sum() - previous_half['Sales'].sum()) / 
                         previous_half['Sales'].sum() * 100) if previous_half['Sales'].sum() > 0 else 0
        
        st.metric("Revenue Growth", f"{revenue_growth:+.1f}%")
        st.metric("Total Customers", f"{filtered_df['Customer ID'].nunique():,}")
    
    with col3:
        # Top performers
        top_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
        top_category = filtered_df.groupby('Product Category')['Profit'].sum().idxmax()
        
        st.info(f"🏆 **Top Region:** {top_region}")
        st.info(f"📦 **Best Category:** {top_category}")
        st.info(f"📊 **Total Orders:** {filtered_df['Order ID'].nunique():,}")
    
    st.markdown("---")
    
    # Business Intelligence Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategic Insights", "📊 Actionable Recommendations", "🚀 Growth Opportunities", "⚠️ Risk Analysis"])
    
    with tab1:
        st.subheader("🎯 Strategic Business Insights")
        
        # Key Business Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Revenue & Profitability Insights")
            
            # Revenue analysis
            revenue_insights = []
            
            # Top revenue drivers
            category_revenue = filtered_df.groupby('Product Category')['Sales'].sum().sort_values(ascending=False)
            region_revenue = filtered_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            
            revenue_insights.append(f"• **{category_revenue.index[0]}** drives {category_revenue.iloc[0]/category_revenue.sum()*100:.1f}% of total revenue")
            revenue_insights.append(f"• **{region_revenue.index[0]}** region contributes {region_revenue.iloc[0]/region_revenue.sum()*100:.1f}% of sales")
            
            # Profit margin analysis
            category_margins = filtered_df.groupby('Product Category').apply(
                lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100
            ).sort_values(ascending=False)
            
            revenue_insights.append(f"• **{category_margins.index[0]}** has the highest profit margin at {category_margins.iloc[0]:.1f}%")
            
            # Customer segment analysis
            segment_revenue = filtered_df.groupby('Customer Segment')['Sales'].sum().sort_values(ascending=False)
            revenue_insights.append(f"• **{segment_revenue.index[0]}** segment generates {segment_revenue.iloc[0]/segment_revenue.sum()*100:.1f}% of revenue")
            
            for insight in revenue_insights:
                st.markdown(insight)
            
            # Revenue concentration chart
            fig = px.pie(values=category_revenue.values, names=category_revenue.index,
                        title="Revenue Concentration by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 👥 Customer Behavior Insights")
            
            customer_insights = []
            
            # Customer analysis
            customer_stats = filtered_df.groupby('Customer ID').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order ID': 'nunique'
            })
            
            avg_customer_value = customer_stats['Sales'].mean()
            repeat_customers = len(customer_stats[customer_stats['Order ID'] > 1])
            total_customers = len(customer_stats)
            repeat_rate = (repeat_customers / total_customers) * 100
            
            customer_insights.append(f"• Average customer lifetime value: **${avg_customer_value:.0f}**")
            customer_insights.append(f"• Repeat customer rate: **{repeat_rate:.1f}%** ({repeat_customers:,} of {total_customers:,})")
            
            # High-value customers
            top_20_pct_threshold = customer_stats['Sales'].quantile(0.8)
            top_customers_revenue = customer_stats[customer_stats['Sales'] >= top_20_pct_threshold]['Sales'].sum()
            top_customers_share = (top_customers_revenue / customer_stats['Sales'].sum()) * 100
            
            customer_insights.append(f"• Top 20% of customers contribute **{top_customers_share:.1f}%** of total revenue")
            
            # Segment preferences
            segment_orders = filtered_df.groupby('Customer Segment')['Order ID'].nunique().sort_values(ascending=False)
            customer_insights.append(f"• **{segment_orders.index[0]}** segment has the most active customers")
            
            for insight in customer_insights:
                st.markdown(insight)
            
            # Customer value distribution
            fig = px.histogram(customer_stats, x='Sales', nbins=30,
                             title="Customer Value Distribution")
            fig.add_vline(x=avg_customer_value, line_dash="dash", 
                         annotation_text="Average", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Market positioning insights
        st.markdown("### 🎯 Market Positioning & Competitive Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Pricing insights
            avg_discount = filtered_df['Discount'].mean()
            high_discount_orders = len(filtered_df[filtered_df['Discount'] > 0.2])
            
            st.markdown("#### 💰 Pricing Strategy")
            st.info(f"Average discount: {avg_discount:.1%}")
            st.info(f"High discount orders (>20%): {high_discount_orders:,}")
            
            if avg_discount > 0.15:
                st.warning("⚠️ High discount dependency detected")
            else:
                st.success("✅ Healthy discount levels")
        
        with col2:
            # Operational efficiency
            avg_delivery = filtered_df['Delivery_Days'].mean()
            shipping_cost_ratio = filtered_df['Shipping Cost'].sum() / filtered_df['Sales'].sum() * 100
            
            st.markdown("#### 🚚 Operational Efficiency")
            st.info(f"Average delivery: {avg_delivery:.1f} days")
            st.info(f"Shipping cost ratio: {shipping_cost_ratio:.1f}%")
            
            if avg_delivery > 5:
                st.warning("⚠️ Delivery times need improvement")
            else:
                st.success("✅ Good delivery performance")
        
        with col3:
            # Product performance
            negative_profit_products = len(filtered_df[filtered_df['Profit'] < 0])
            product_efficiency = (1 - negative_profit_products / len(filtered_df)) * 100
            
            st.markdown("#### 📦 Product Portfolio")
            st.info(f"Product efficiency: {product_efficiency:.1f}%")
            st.info(f"Loss-making orders: {negative_profit_products:,}")
            
            if product_efficiency < 90:
                st.warning("⚠️ Product portfolio needs optimization")
            else:
                st.success("✅ Strong product performance")
    
    with tab2:
        st.subheader("📊 Actionable Business Recommendations")
        
        # Priority recommendations based on data analysis
        recommendations = []
        
        # Revenue optimization recommendations
        st.markdown("### 🎯 Priority Action Items")
        
        # Analyze key metrics for recommendations
        profit_margin = (filtered_df['Profit'].sum() / filtered_df['Sales'].sum()) * 100
        avg_discount = filtered_df['Discount'].mean()
        loss_making_orders = len(filtered_df[filtered_df['Profit'] < 0])
        
        # Category performance
        category_performance = filtered_df.groupby('Product Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        })
        category_performance['Margin'] = (category_performance['Profit'] / category_performance['Sales']) * 100
        
        worst_category = category_performance['Margin'].idxmin()
        best_category = category_performance['Margin'].idxmax()
        
        # Regional performance
        region_performance = filtered_df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        })
        region_performance['Profit_per_Sale'] = region_performance['Profit'] / region_performance['Sales']
        
        best_region = region_performance['Profit_per_Sale'].idxmax()
        worst_region = region_performance['Profit_per_Sale'].idxmin()
        
        # Generate recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Immediate Actions (High Priority)")
            
            immediate_actions = []
            
            if profit_margin < 10:
                immediate_actions.append({
                    "title": "🔴 Critical: Improve Profit Margins",
                    "description": f"Current margin of {profit_margin:.1f}% is below healthy threshold",
                    "actions": [
                        "Review pricing strategy immediately",
                        "Reduce operational costs",
                        "Focus on high-margin products"
                    ],
                    "impact": "High",
                    "timeline": "1-2 weeks"
                })
            
            if loss_making_orders > len(filtered_df) * 0.1:
                immediate_actions.append({
                    "title": "🔴 Critical: Address Loss-Making Orders",
                    "description": f"{loss_making_orders} orders ({loss_making_orders/len(filtered_df)*100:.1f}%) are generating losses",
                    "actions": [
                        f"Review {worst_category} category pricing",
                        "Discontinue unprofitable products",
                        "Renegotiate supplier costs"
                    ],
                    "impact": "High",
                    "timeline": "2-4 weeks"
                })
            
            if avg_discount > 0.15:
                immediate_actions.append({
                    "title": "🟡 Review Discount Strategy",
                    "description": f"Average discount of {avg_discount:.1%} may be eroding profits",
                    "actions": [
                        "Implement dynamic pricing",
                        "Create value-based packages",
                        "Reduce dependency on discounting"
                    ],
                    "impact": "Medium",
                    "timeline": "3-6 weeks"
                })
            
            for i, action in enumerate(immediate_actions, 1):
                with st.expander(f"{i}. {action['title']}", expanded=True):
                    st.write(action['description'])
                    st.write("**Recommended Actions:**")
                    for rec_action in action['actions']:
                        st.write(f"• {rec_action}")
                    st.write(f"**Impact:** {action['impact']} | **Timeline:** {action['timeline']}")
        
        with col2:
            st.markdown("#### 📈 Growth Opportunities (Medium Priority)")
            
            growth_opportunities = []
            
            # Regional expansion
            if len(region_performance) > 1:
                growth_opportunities.append({
                    "title": f"🌍 Expand {best_region} Region Success",
                    "description": f"{best_region} shows highest profitability per sale",
                    "actions": [
                        f"Replicate {best_region} strategies in other regions",
                        "Increase marketing investment in top-performing regions",
                        "Analyze regional customer preferences"
                    ],
                    "potential": f"+{region_performance.loc[best_region, 'Sales'] * 0.2:,.0f} revenue"
                })
            
            # Customer segment focus
            top_segment = filtered_df.groupby('Customer Segment')['Profit'].sum().idxmax()
            growth_opportunities.append({
                "title": f"👥 Focus on {top_segment} Segment",
                "description": f"{top_segment} segment shows highest profitability",
                "actions": [
                    f"Develop {top_segment}-specific marketing campaigns",
                    "Create loyalty programs for high-value segments",
                    "Analyze segment-specific product preferences"
                ],
                "potential": "15-25% revenue increase"
            })
            
            # Product optimization
            growth_opportunities.append({
                "title": f"📦 Optimize {best_category} Category",
                "description": f"{best_category} shows best margins - expand offerings",
                "actions": [
                    f"Increase {best_category} product range",
                    "Bundle high-margin products",
                    "Cross-sell complementary items"
                ],
                "potential": f"+{category_performance.loc[best_category, 'Sales'] * 0.3:,.0f} revenue"
            })
            
            for i, opportunity in enumerate(growth_opportunities, 1):
                with st.expander(f"{i}. {opportunity['title']}", expanded=False):
                    st.write(opportunity['description'])
                    st.write("**Growth Actions:**")
                    for growth_action in opportunity['actions']:
                        st.write(f"• {growth_action}")
                    st.write(f"**Potential Impact:** {opportunity['potential']}")
        
        # Implementation roadmap
        st.markdown("### 🗓️ Implementation Roadmap")
        
        roadmap_data = {
            "Week 1-2": ["Review pricing strategy", "Identify loss-making products", "Analyze discount patterns"],
            "Week 3-4": ["Implement pricing changes", "Discontinue unprofitable items", "Launch targeted campaigns"],
            "Week 5-8": ["Monitor performance changes", "Expand successful strategies", "Optimize operations"],
            "Week 9-12": ["Scale proven initiatives", "Develop new products", "Enter new markets"]
        }
        
        for timeline, tasks in roadmap_data.items():
            st.markdown(f"**{timeline}:**")
            for task in tasks:
                st.markdown(f"  • {task}")
    
    with tab3:
        st.subheader("🚀 Growth Opportunities & Market Expansion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Market Expansion Analysis")
            
            # Identify expansion opportunities
            state_performance = filtered_df.groupby('State or Province').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Customer ID': 'nunique'
            }).sort_values('Sales', ascending=False)
            
            # Top growth markets
            st.markdown("#### 🎯 Top Growth Markets")
            top_states = state_performance.head(5)
            
            for state in top_states.index:
                sales = top_states.loc[state, 'Sales']
                customers = top_states.loc[state, 'Customer ID']
                revenue_per_customer = sales / customers
                
                st.info(f"**{state}**: ${sales:,.0f} sales, {customers} customers (${revenue_per_customer:.0f}/customer)")
            
            # Underperforming markets with potential
            st.markdown("#### 🌱 Untapped Potential Markets")
            bottom_states = state_performance.tail(5)
            
            for state in bottom_states.index:
                sales = bottom_states.loc[state, 'Sales']
                customers = bottom_states.loc[state, 'Customer ID']
                
                st.warning(f"**{state}**: ${sales:,.0f} sales, {customers} customers - Growth opportunity")
        
        with col2:
            st.markdown("### 💡 Product Innovation Opportunities")
            
            # Product mix analysis
            product_performance = filtered_df.groupby('Product Sub-Category').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Quantity ordered new': 'sum'
            }).sort_values('Profit', ascending=False)
            
            # High-margin, low-volume products
            product_performance['Profit_Margin'] = (product_performance['Profit'] / product_performance['Sales']) * 100
            high_margin_products = product_performance[product_performance['Profit_Margin'] > 20].head(5)
            
            st.markdown("#### 🏆 High-Margin Products to Scale")
            for product in high_margin_products.index:
                margin = high_margin_products.loc[product, 'Profit_Margin']
                quantity = high_margin_products.loc[product, 'Quantity ordered new']
                
                st.success(f"**{product}**: {margin:.1f}% margin, {quantity} units sold")
            
            # Cross-selling opportunities
            st.markdown("#### 🔗 Cross-Selling Opportunities")
            
            # Analyze customer purchase patterns
            customer_categories = filtered_df.groupby('Customer ID')['Product Category'].nunique()
            single_category_customers = len(customer_categories[customer_categories == 1])
            total_customers = len(customer_categories)
            cross_sell_potential = (single_category_customers / total_customers) * 100
            
            st.info(f"**Cross-sell Potential**: {cross_sell_potential:.1f}% of customers buy from only one category")
            st.info(f"**Opportunity**: {single_category_customers:,} customers could buy additional categories")
        
        # Growth projections
        st.markdown("### 📈 Growth Projections & Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        current_monthly_revenue = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().mean()
        
        with col1:
            st.markdown("#### 🐌 Conservative Growth (5%)")
            conservative_growth = current_monthly_revenue * 1.05 * 12
            st.metric("Annual Revenue Projection", f"${conservative_growth:,.0f}")
            st.info("Focus on operational efficiency")
        
        with col2:
            st.markdown("#### 📈 Moderate Growth (15%)")
            moderate_growth = current_monthly_revenue * 1.15 * 12
            st.metric("Annual Revenue Projection", f"${moderate_growth:,.0f}")
            st.info("Market expansion + optimization")
        
        with col3:
            st.markdown("#### 🚀 Aggressive Growth (30%)")
            aggressive_growth = current_monthly_revenue * 1.30 * 12
            st.metric("Annual Revenue Projection", f"${aggressive_growth:,.0f}")
            st.info("New markets + products + segments")
        
        # Investment requirements
        st.markdown("### 💰 Investment Requirements")
        
        investment_scenarios = {
            "Conservative": {"Marketing": 50000, "Operations": 30000, "Technology": 20000},
            "Moderate": {"Marketing": 100000, "Operations": 75000, "Technology": 50000},
            "Aggressive": {"Marketing": 200000, "Operations": 150000, "Technology": 100000}
        }
        
        scenario_df = pd.DataFrame(investment_scenarios).T
        st.dataframe(scenario_df, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Risk Analysis & Mitigation")
        
        # Risk assessment
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 Identified Business Risks")
            
            risks = []
            
            # Financial risks
            if profit_margin < 10:
                risks.append({
                    "type": "Financial",
                    "risk": "Low Profit Margins",
                    "severity": "High",
                    "impact": f"Current margin: {profit_margin:.1f}%",
                    "mitigation": ["Increase prices", "Reduce costs", "Focus on high-margin products"]
                })
            
            # Customer concentration risk
            top_customers = filtered_df.groupby('Customer ID')['Sales'].sum().sort_values(ascending=False)
            top_10_share = top_customers.head(10).sum() / top_customers.sum() * 100
            
            if top_10_share > 30:
                risks.append({
                    "type": "Customer",
                    "risk": "Customer Concentration",
                    "severity": "Medium",
                    "impact": f"Top 10 customers: {top_10_share:.1f}% of revenue",
                    "mitigation": ["Diversify customer base", "Reduce dependency", "Improve retention"]
                })
            
            # Geographic concentration
            region_concentration = filtered_df.groupby('Region')['Sales'].sum()
            top_region_share = region_concentration.max() / region_concentration.sum() * 100
            
            if top_region_share > 40:
                risks.append({
                    "type": "Geographic",
                    "risk": "Regional Concentration",
                    "severity": "Medium",
                    "impact": f"Top region: {top_region_share:.1f}% of sales",
                    "mitigation": ["Expand to new regions", "Diversify markets", "Local partnerships"]
                })
            
            # Operational risks
            avg_delivery = filtered_df['Delivery_Days'].mean()
            if avg_delivery > 5:
                risks.append({
                    "type": "Operational",
                    "risk": "Slow Delivery Times",
                    "severity": "Medium",
                    "impact": f"Average delivery: {avg_delivery:.1f} days",
                    "mitigation": ["Optimize logistics", "Partner with faster carriers", "Improve forecasting"]
                })
            
            for i, risk in enumerate(risks, 1):
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                
                with st.expander(f"{severity_color[risk['severity']]} {i}. {risk['risk']} ({risk['type']})", expanded=True):
                    st.write(f"**Severity:** {risk['severity']}")
                    st.write(f"**Impact:** {risk['impact']}")
                    st.write("**Mitigation Strategies:**")
                    for mitigation in risk['mitigation']:
                        st.write(f"• {mitigation}")
        
        with col2:
            st.markdown("### 📊 Risk Monitoring Dashboard")
            
            # Key risk metrics
            risk_metrics = {
                "Profit Margin": {"value": profit_margin, "threshold": 10, "status": "danger" if profit_margin < 10 else "success"},
                "Customer Concentration": {"value": top_10_share, "threshold": 30, "status": "warning" if top_10_share > 30 else "success"},
                "Loss-Making Orders": {"value": loss_making_orders/len(filtered_df)*100, "threshold": 10, "status": "danger" if loss_making_orders/len(filtered_df)*100 > 10 else "success"},
                "Average Delivery Days": {"value": avg_delivery, "threshold": 5, "status": "warning" if avg_delivery > 5 else "success"}
            }
            
            for metric, data in risk_metrics.items():
                if data['status'] == 'danger':
                    st.error(f"🔴 **{metric}**: {data['value']:.1f}% (Threshold: {data['threshold']}%)")
                elif data['status'] == 'warning':
                    st.warning(f"🟡 **{metric}**: {data['value']:.1f}% (Threshold: {data['threshold']}%)")
                else:
                    st.success(f"🟢 **{metric}**: {data['value']:.1f}% (Threshold: {data['threshold']}%)")
            
            # Risk trend analysis
            monthly_profit_margin = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).apply(
                lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100
            )
            
            fig = px.line(x=monthly_profit_margin.index.astype(str), y=monthly_profit_margin.values,
                         title="Monthly Profit Margin Trend")
            fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk mitigation action plan
        st.markdown("### 🛡️ Risk Mitigation Action Plan")
        
        mitigation_plan = {
            "Immediate (1-2 weeks)": [
                "Monitor daily profit margins",
                "Identify top 10 at-risk customers",
                "Review pricing for loss-making products"
            ],
            "Short-term (1-3 months)": [
                "Implement customer diversification strategy",
                "Launch retention campaigns for key customers",
                "Optimize operational efficiency"
            ],
            "Long-term (3-12 months)": [
                "Expand to new geographic markets",
                "Develop alternative revenue streams",
                "Build strategic partnerships"
            ]
        }
        
        for timeline, actions in mitigation_plan.items():
            st.markdown(f"#### {timeline}")
            for action in actions:
                st.markdown(f"• {action}")
    
    # Overall business score
    st.markdown("---")
    st.subheader("🎯 Overall Business Performance Score")
    
    # Calculate composite score
    scores = {
        "Profitability": min(100, (profit_margin / 15) * 100),
        "Growth": min(100, (revenue_growth + 10) * 5) if revenue_growth > -10 else 0,
        "Customer Health": min(100, (repeat_rate / 50) * 100),
        "Operational Efficiency": min(100, max(0, (10 - avg_delivery) / 10 * 100))
    }
    
    overall_score = sum(scores.values()) / len(scores)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Profitability Score", f"{scores['Profitability']:.0f}/100")
    with col2:
        st.metric("Growth Score", f"{scores['Growth']:.0f}/100")
    with col3:
        st.metric("Customer Score", f"{scores['Customer Health']:.0f}/100")
    with col4:
        st.metric("Operations Score", f"{scores['Operational Efficiency']:.0f}/100")
    with col5:
        if overall_score >= 80:
            st.success(f"🏆 Overall Score: {overall_score:.0f}/100")
        elif overall_score >= 60:
            st.warning(f"⚠️ Overall Score: {overall_score:.0f}/100")
        else:
            st.error(f"🚨 Overall Score: {overall_score:.0f}/100")

# Placeholder for remaining pages - we'll build these in the next steps
# ============================================================================
# PAGE 5: DETAILED REPORTS
# ============================================================================
elif page == "📋 Detailed Reports":
   st.markdown('<h1 class="main-header">📋 Detailed Reports & Data Export</h1>', unsafe_allow_html=True)
   
   # Report generation options
   st.subheader("📊 Report Generation Options")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       report_type = st.selectbox(
           "Select Report Type",
           ["Sales Report", "Profit Analysis", "Customer Report", "Product Report", "Regional Report", "Custom Report"]
       )
   
   with col2:
       report_format = st.selectbox(
           "Select Format",
           ["Table View", "Summary Statistics", "Detailed Analysis"]
       )
   
   with col3:
       export_format = st.selectbox(
           "Export Format",
           ["CSV", "Excel", "PDF Summary"]
       )
   
   # Report tabs
   tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Sales Reports", "💰 Financial Reports", "👥 Customer Reports", "📦 Product Reports", "🌍 Regional Reports"])
   
   with tab1:
       st.subheader("📈 Sales Performance Reports")
       
       # Sales report options
       col1, col2 = st.columns([1, 3])
       
       with col1:
           st.markdown("#### Report Options")
           
           # Grouping options
           group_by = st.selectbox(
               "Group By",
               ["Date", "Product Category", "Customer Segment", "Region", "State", "Ship Mode"]
           )
           
           # Aggregation options
           metrics = st.multiselect(
               "Select Metrics",
               ["Sales", "Quantity", "Orders", "Average Order Value", "Customers"],
               default=["Sales", "Orders"]
           )
           
           # Date granularity
           if group_by == "Date":
               date_granularity = st.selectbox(
                   "Date Granularity",
                   ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
               )
       
       with col2:
           st.markdown(f"#### Sales Report - Grouped by {group_by}")
           
           # Generate report based on selections
           if group_by == "Date":
               if date_granularity == "Daily":
                   grouped_data = filtered_df.groupby('Order Date')
               elif date_granularity == "Weekly":
                   grouped_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('W'))
               elif date_granularity == "Monthly":
                   grouped_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))
               elif date_granularity == "Quarterly":
                   grouped_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('Q'))
               else:  # Yearly
                   grouped_data = filtered_df.groupby(filtered_df['Order Date'].dt.year)
           else:
               grouped_data = filtered_df.groupby(group_by)
           
           # Calculate metrics
           report_data = {}
           
           if "Sales" in metrics:
               report_data["Total Sales"] = grouped_data['Sales'].sum()
           if "Quantity" in metrics:
               report_data["Total Quantity"] = grouped_data['Quantity ordered new'].sum()
           if "Orders" in metrics:
               report_data["Total Orders"] = grouped_data['Order ID'].nunique()
           if "Average Order Value" in metrics:
               report_data["Avg Order Value"] = grouped_data['Sales'].mean()
           if "Customers" in metrics:
               report_data["Unique Customers"] = grouped_data['Customer ID'].nunique()
           
           sales_report_df = pd.DataFrame(report_data).round(2)
           
           # Add percentage calculations
           if "Sales" in metrics:
               sales_report_df["Sales %"] = (sales_report_df["Total Sales"] / sales_report_df["Total Sales"].sum() * 100).round(1)
           
           # Sort by total sales if available
           if "Total Sales" in sales_report_df.columns:
               sales_report_df = sales_report_df.sort_values("Total Sales", ascending=False)
           
           st.dataframe(sales_report_df, use_container_width=True)
           
           # Summary statistics
           if not sales_report_df.empty:
               st.markdown("#### 📊 Summary Statistics")
               
               col1, col2, col3, col4 = st.columns(4)
               
               with col1:
                   if "Total Sales" in sales_report_df.columns:
                       st.metric("Total Sales", f"${sales_report_df['Total Sales'].sum():,.0f}")
               
               with col2:
                   if "Total Orders" in sales_report_df.columns:
                       st.metric("Total Orders", f"{sales_report_df['Total Orders'].sum():,}")
               
               with col3:
                   if "Unique Customers" in sales_report_df.columns:
                       st.metric("Total Customers", f"{sales_report_df['Unique Customers'].sum():,}")
               
               with col4:
                   if "Avg Order Value" in sales_report_df.columns:
                       st.metric("Overall AOV", f"${sales_report_df['Avg Order Value'].mean():.0f}")
           
           # Download button
           if not sales_report_df.empty:
               csv_data = sales_report_df.to_csv(index=True)
               st.download_button(
                   label="📥 Download Sales Report",
                   data=csv_data,
                   file_name=f"sales_report_{group_by.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                   mime="text/csv"
               )
   
   with tab2:
       st.subheader("💰 Financial Performance Reports")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           st.markdown("#### Financial Metrics")
           
           financial_metrics = st.multiselect(
               "Select Financial Metrics",
               ["Revenue", "Profit", "Profit Margin", "Shipping Cost", "Discount", "ROI"],
               default=["Revenue", "Profit", "Profit Margin"]
           )
           
           financial_group = st.selectbox(
               "Group By",
               ["Product Category", "Customer Segment", "Region", "Month", "Quarter"],
               key="financial_group"
           )
       
       with col2:
           st.markdown(f"#### Financial Report - {financial_group}")
           
           # Generate financial report
           if financial_group == "Month":
               fin_grouped = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))
           elif financial_group == "Quarter":
               fin_grouped = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('Q'))
           else:
               fin_grouped = filtered_df.groupby(financial_group)
           
           fin_report_data = {}
           
           if "Revenue" in financial_metrics:
               fin_report_data["Revenue"] = fin_grouped['Sales'].sum()
           if "Profit" in financial_metrics:
               fin_report_data["Profit"] = fin_grouped['Profit'].sum()
           if "Profit Margin" in financial_metrics:
               fin_report_data["Profit Margin %"] = (fin_grouped['Profit'].sum() / fin_grouped['Sales'].sum() * 100)
           if "Shipping Cost" in financial_metrics:
               fin_report_data["Shipping Cost"] = fin_grouped['Shipping Cost'].sum()
           if "Discount" in financial_metrics:
               fin_report_data["Total Discount"] = fin_grouped['Discount'].sum()
           if "ROI" in financial_metrics:
               cost_base = fin_grouped['Sales'].sum() - fin_grouped['Profit'].sum()
               fin_report_data["ROI %"] = (fin_grouped['Profit'].sum() / cost_base * 100).fillna(0)
           
           financial_report_df = pd.DataFrame(fin_report_data).round(2)
           
           if not financial_report_df.empty:
               # Sort by profit if available
               if "Profit" in financial_report_df.columns:
                   financial_report_df = financial_report_df.sort_values("Profit", ascending=False)
               
               st.dataframe(financial_report_df, use_container_width=True)
               
               # Financial KPIs
               st.markdown("#### 💼 Key Financial KPIs")
               
               col1, col2, col3, col4 = st.columns(4)
               
               with col1:
                   if "Revenue" in financial_report_df.columns:
                       total_revenue = financial_report_df["Revenue"].sum()
                       st.metric("Total Revenue", f"${total_revenue:,.0f}")
               
               with col2:
                   if "Profit" in financial_report_df.columns:
                       total_profit = financial_report_df["Profit"].sum()
                       st.metric("Total Profit", f"${total_profit:,.0f}")
               
               with col3:
                   if "Profit Margin %" in financial_report_df.columns:
                       avg_margin = financial_report_df["Profit Margin %"].mean()
                       st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
               
               with col4:
                   if "ROI %" in financial_report_df.columns:
                       avg_roi = financial_report_df["ROI %"].mean()
                       st.metric("Average ROI", f"{avg_roi:.1f}%")
               
               # Profitability analysis
               if "Profit" in financial_report_df.columns:
                   profitable_count = len(financial_report_df[financial_report_df["Profit"] > 0])
                   total_count = len(financial_report_df)
                   profitability_rate = (profitable_count / total_count) * 100
                   
                   if profitability_rate >= 90:
                       st.success(f"✅ Profitability Rate: {profitability_rate:.1f}% ({profitable_count}/{total_count})")
                   elif profitability_rate >= 75:
                       st.warning(f"⚠️ Profitability Rate: {profitability_rate:.1f}% ({profitable_count}/{total_count})")
                   else:
                       st.error(f"🚨 Profitability Rate: {profitability_rate:.1f}% ({profitable_count}/{total_count})")
               
               # Download financial report
               csv_data = financial_report_df.to_csv(index=True)
               st.download_button(
                   label="📥 Download Financial Report",
                   data=csv_data,
                   file_name=f"financial_report_{financial_group.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                   mime="text/csv"
               )
   
   with tab3:
       st.subheader("👥 Customer Analysis Reports")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           st.markdown("#### Customer Metrics")
           
           customer_analysis_type = st.selectbox(
               "Analysis Type",
               ["Customer Lifetime Value", "Purchase Behavior", "Segmentation Analysis", "Retention Analysis"]
           )
           
           customer_metrics = st.multiselect(
               "Customer Metrics",
               ["Total Spent", "Orders Count", "Avg Order Value", "Profit Generated", "First Purchase", "Last Purchase"],
               default=["Total Spent", "Orders Count", "Avg Order Value"]
           )
       
       with col2:
           st.markdown(f"#### {customer_analysis_type}")
           
           # Customer analysis
           customer_analysis = filtered_df.groupby('Customer ID').agg({
               'Sales': ['sum', 'mean', 'count'],
               'Profit': 'sum',
               'Order Date': ['min', 'max'],
               'Order ID': 'nunique'
           }).round(2)
           
           # Flatten column names
           customer_analysis.columns = ['Total_Spent', 'Avg_Order_Value', 'Total_Orders', 'Profit_Generated', 'First_Purchase', 'Last_Purchase', 'Unique_Orders']
           
           # Add customer names
           customer_names = filtered_df.groupby('Customer ID')['Customer Name'].first()
           customer_analysis['Customer_Name'] = customer_names
           
           # Reorder columns
           customer_analysis = customer_analysis[['Customer_Name'] + [col for col in customer_analysis.columns if col != 'Customer_Name']]
           
           if customer_analysis_type == "Customer Lifetime Value":
               # Sort by total spent
               customer_clv = customer_analysis.sort_values('Total_Spent', ascending=False)
               
               # Add CLV tiers
               customer_clv['CLV_Tier'] = pd.qcut(customer_clv['Total_Spent'], 
                                                 q=5, 
                                                 labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'])
               
               st.dataframe(customer_clv.head(20), use_container_width=True)
               
               # CLV distribution
               col1, col2 = st.columns(2)
               
               with col1:
                   tier_counts = customer_clv['CLV_Tier'].value_counts()
                   fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                              title="Customer Value Distribution")
                   st.plotly_chart(fig, use_container_width=True)
               
               with col2:
                   # Top customers
                   st.markdown("##### 🏆 Top 10 Customers")
                   top_customers = customer_clv.head(10)[['Customer_Name', 'Total_Spent', 'Total_Orders']]
                   st.dataframe(top_customers, hide_index=True, use_container_width=True)
           
           elif customer_analysis_type == "Purchase Behavior":
               # Purchase frequency analysis
               customer_analysis['Days_Between_Orders'] = (
                   (customer_analysis['Last_Purchase'] - customer_analysis['First_Purchase']).dt.days / 
                   customer_analysis['Total_Orders'].clip(lower=1)
               ).fillna(0)
               
               customer_analysis['Purchase_Frequency'] = np.where(
                   customer_analysis['Days_Between_Orders'] <= 30, 'High',
                   np.where(customer_analysis['Days_Between_Orders'] <= 90, 'Medium', 'Low')
               )
               
               behavior_analysis = customer_analysis.sort_values('Total_Orders', ascending=False)
               st.dataframe(behavior_analysis.head(20), use_container_width=True)
               
               # Behavior insights
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   high_freq = len(behavior_analysis[behavior_analysis['Purchase_Frequency'] == 'High'])
                   st.metric("High Frequency Customers", high_freq)
               
               with col2:
                   avg_days_between = behavior_analysis['Days_Between_Orders'].mean()
                   st.metric("Avg Days Between Orders", f"{avg_days_between:.0f}")
               
               with col3:
                   repeat_customers = len(behavior_analysis[behavior_analysis['Total_Orders'] > 1])
                   st.metric("Repeat Customers", repeat_customers)
           
           elif customer_analysis_type == "Segmentation Analysis":
               # Customer segmentation by segment
               segment_analysis = filtered_df.groupby(['Customer Segment', 'Customer ID']).agg({
                   'Sales': 'sum',
                   'Profit': 'sum',
                   'Order ID': 'nunique'
               }).reset_index()
               
               segment_summary = segment_analysis.groupby('Customer Segment').agg({
                   'Sales': ['mean', 'median', 'sum'],
                   'Profit': ['mean', 'sum'],
                   'Order ID': ['mean', 'sum'],
                   'Customer ID': 'count'
               }).round(2)
               
               segment_summary.columns = ['Avg_Sales', 'Median_Sales', 'Total_Sales', 'Avg_Profit', 'Total_Profit', 'Avg_Orders', 'Total_Orders', 'Customer_Count']
               
               st.dataframe(segment_summary, use_container_width=True)
               
               # Segment performance chart
               fig = px.scatter(segment_summary, x='Avg_Sales', y='Avg_Profit',
                              size='Customer_Count', hover_name=segment_summary.index,
                              title="Customer Segment Performance")
               st.plotly_chart(fig, use_container_width=True)
           
           else:  # Retention Analysis
               # Calculate customer retention metrics
               customer_analysis['Days_Since_Last_Purchase'] = (
                   datetime.now() - customer_analysis['Last_Purchase']
               ).dt.days
               
               customer_analysis['Customer_Status'] = np.where(
                   customer_analysis['Days_Since_Last_Purchase'] <= 90, 'Active',
                   np.where(customer_analysis['Days_Since_Last_Purchase'] <= 180, 'At Risk', 'Churned')
               )
               
               retention_analysis = customer_analysis.sort_values('Days_Since_Last_Purchase')
               st.dataframe(retention_analysis.head(20), use_container_width=True)
               
               # Retention metrics
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   active_customers = len(retention_analysis[retention_analysis['Customer_Status'] == 'Active'])
                   st.metric("Active Customers", active_customers)
               
               with col2:
                   at_risk_customers = len(retention_analysis[retention_analysis['Customer_Status'] == 'At Risk'])
                   st.metric("At Risk Customers", at_risk_customers)
               
               with col3:
                   churned_customers = len(retention_analysis[retention_analysis['Customer_Status'] == 'Churned'])
                   st.metric("Churned Customers", churned_customers)
           
           # Download customer report
           csv_data = customer_analysis.to_csv(index=True)
           st.download_button(
               label="📥 Download Customer Report",
               data=csv_data,
               file_name=f"customer_report_{customer_analysis_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
               mime="text/csv"
           )
   
   with tab4:
       st.subheader("📦 Product Performance Reports")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           st.markdown("#### Product Analysis")
           
           product_level = st.selectbox(
               "Product Level",
               ["Product Category", "Product Sub-Category", "Product Name"]
           )
           
           product_metrics = st.multiselect(
               "Product Metrics",
               ["Sales Revenue", "Profit", "Quantity Sold", "Profit Margin", "Units per Order", "Return Rate"],
               default=["Sales Revenue", "Profit", "Quantity Sold"]
           )
           
           sort_by = st.selectbox(
               "Sort By",
               ["Sales Revenue", "Profit", "Quantity Sold", "Profit Margin"]
           )
       
       with col2:
           st.markdown(f"#### Product Analysis - {product_level}")
           
           # Product analysis
           product_analysis = filtered_df.groupby(product_level).agg({
               'Sales': 'sum',
               'Profit': 'sum',
               'Quantity ordered new': 'sum',
               'Order ID': 'nunique',
               'Discount': 'mean'
           }).round(2)
           
           # Calculate additional metrics
           product_analysis['Profit_Margin'] = (product_analysis['Profit'] / product_analysis['Sales'] * 100).round(2)
           product_analysis['Avg_Units_per_Order'] = (product_analysis['Quantity ordered new'] / product_analysis['Order ID']).round(2)
           product_analysis['Avg_Discount'] = (product_analysis['Discount'] * 100).round(2)
           
           # Rename columns for display
           column_mapping = {
               'Sales': 'Sales Revenue',
               'Profit': 'Profit',
               'Quantity ordered new': 'Quantity Sold',
               'Order ID': 'Total Orders',
               'Profit_Margin': 'Profit Margin %',
               'Avg_Units_per_Order': 'Units per Order',
               'Avg_Discount': 'Avg Discount %'
           }
           
           product_analysis = product_analysis.rename(columns=column_mapping)
           
           # Sort by selected metric
           if sort_by in product_analysis.columns:
               product_analysis = product_analysis.sort_values(sort_by, ascending=False)
           
           st.dataframe(product_analysis, use_container_width=True)
           
           # Product insights
           st.markdown("#### 📊 Product Performance Insights")
           
           col1, col2, col3, col4 = st.columns(4)
           
           with col1:
               top_seller = product_analysis['Sales Revenue'].idxmax()
               top_sales = product_analysis.loc[top_seller, 'Sales Revenue']
               st.success(f"🏆 **Top Seller**\n{top_seller}\n${top_sales:,.0f}")
           
           with col2:
               most_profitable = product_analysis['Profit'].idxmax()
               top_profit = product_analysis.loc[most_profitable, 'Profit']
               st.success(f"💰 **Most Profitable**\n{most_profitable}\n${top_profit:,.0f}")
           
           with col3:
               highest_margin = product_analysis['Profit Margin %'].idxmax()
               margin_value = product_analysis.loc[highest_margin, 'Profit Margin %']
               st.success(f"📈 **Highest Margin**\n{highest_margin}\n{margin_value:.1f}%")
           
           with col4:
               most_volume = product_analysis['Quantity Sold'].idxmax()
               volume_value = product_analysis.loc[most_volume, 'Quantity Sold']
               st.success(f"📦 **Highest Volume**\n{most_volume}\n{volume_value:,.0f} units")
           
           # Performance categories
           st.markdown("#### 🎯 Product Performance Categories")
           
           # Categorize products
           high_performers = product_analysis[
               (product_analysis['Sales Revenue'] > product_analysis['Sales Revenue'].quantile(0.8)) &
               (product_analysis['Profit Margin %'] > product_analysis['Profit Margin %'].median())
           ]
           
           underperformers = product_analysis[
               (product_analysis['Sales Revenue'] < product_analysis['Sales Revenue'].quantile(0.2)) |
               (product_analysis['Profit Margin %'] < 0)
           ]
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("##### 🌟 High Performers (High Sales + Good Margins)")
               if not high_performers.empty:
                   st.dataframe(high_performers[['Sales Revenue', 'Profit', 'Profit Margin %']].head(10), use_container_width=True)
               else:
                   st.info("No products meet high performer criteria")
           
           with col2:
               st.markdown("##### ⚠️ Underperformers (Low Sales or Negative Margins)")
               if not underperformers.empty:
                   st.dataframe(underperformers[['Sales Revenue', 'Profit', 'Profit Margin %']].head(10), use_container_width=True)
               else:
                   st.success("No underperforming products found")
           
           # Product performance visualization
           if len(product_analysis) <= 20:  # Only show chart if not too many products
               fig = px.scatter(product_analysis, x='Sales Revenue', y='Profit Margin %',
                              size='Quantity Sold', hover_name=product_analysis.index,
                              title=f"{product_level} Performance: Sales vs Profit Margin")
               st.plotly_chart(fig, use_container_width=True)
           
           # Download product report
           csv_data = product_analysis.to_csv(index=True)
           st.download_button(
               label="📥 Download Product Report",
               data=csv_data,
               file_name=f"product_report_{product_level.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
               mime="text/csv"
           )
   
   with tab5:
       st.subheader("🌍 Regional Performance Reports")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           st.markdown("#### Regional Analysis")
           
           geographic_level = st.selectbox(
               "Geographic Level",
               ["Region", "State or Province", "City"]
           )
           
           regional_metrics = st.multiselect(
               "Regional Metrics",
               ["Sales", "Profit", "Orders", "Customers", "Market Share", "Growth Rate"],
               default=["Sales", "Profit", "Orders", "Customers"]
           )
       
       with col2:
           st.markdown(f"#### Regional Report - {geographic_level}")
           
           # Regional analysis
           regional_analysis = filtered_df.groupby(geographic_level).agg({
               'Sales': 'sum',
               'Profit': 'sum',
               'Order ID': 'nunique',
               'Customer ID': 'nunique',
               'Shipping Cost': 'sum',
               'Delivery_Days': 'mean'
           }).round(2)
           
           # Calculate additional metrics
           regional_analysis['Profit_Margin'] = (regional_analysis['Profit'] / regional_analysis['Sales'] * 100).round(2)
           regional_analysis['Sales_per_Customer'] = (regional_analysis['Sales'] / regional_analysis['Customer ID']).round(2)
           regional_analysis['Orders_per_Customer'] = (regional_analysis['Order ID'] / regional_analysis['Customer ID']).round(2)
           regional_analysis['Market_Share'] = (regional_analysis['Sales'] / regional_analysis['Sales'].sum() * 100).round(2)
           
           # Rename columns
           regional_analysis = regional_analysis.rename(columns={
               'Sales': 'Total Sales',
               'Profit': 'Total Profit',
               'Order ID': 'Total Orders',
               'Customer ID': 'Total Customers',
               'Shipping Cost': 'Total Shipping Cost',
               'Delivery_Days': 'Avg Delivery Days',
               'Profit_Margin': 'Profit Margin %',
               'Sales_per_Customer': 'Sales per Customer',
               'Orders_per_Customer': 'Orders per Customer',
               'Market_Share': 'Market Share %'
           })
           
           # Sort by total sales
           regional_analysis = regional_analysis.sort_values('Total Sales', ascending=False)
           
           st.dataframe(regional_analysis, use_container_width=True)
           
           # Regional insights
           st.markdown("#### 🗺️ Regional Performance Insights")
           
           col1, col2, col3, col4 = st.columns(4)
           
           with col1:
               top_region = regional_analysis['Total Sales'].idxmax()
               top_sales = regional_analysis.loc[top_region, 'Total Sales']
               market_share = regional_analysis.loc[top_region, 'Market Share %']
               st.success(f"🏆 **Top Region**\n{top_region}\n${top_sales:,.0f} ({market_share:.1f}%)")
           
           with col2:
               most_profitable_region = regional_analysis['Total Profit'].idxmax()
               top_profit = regional_analysis.loc[most_profitable_region, 'Total Profit']
               st.success(f"💰 **Most Profitable**\n{most_profitable_region}\n${top_profit:,.0f}")
           
           with col3:
               highest_margin_region = regional_analysis['Profit Margin %'].idxmax()
               margin = regional_analysis.loc[highest_margin_region, 'Profit Margin %']
               st.success(f"📈 **Best Margin**\n{highest_margin_region}\n{margin:.1f}%")
           
           with col4:
               most_customers_region = regional_analysis['Total Customers'].idxmax()
               customer_count = regional_analysis.loc[most_customers_region, 'Total Customers']
               st.success(f"👥 **Most Customers**\n{most_customers_region}\n{customer_count:,}")
           
           # Regional performance charts
           col1, col2 = st.columns(2)
           
           with col1:
               # Top regions by sales
               top_regions = regional_analysis.head(10)
               fig = px.bar(x=top_regions.index, y=top_regions['Total Sales'],
                          title=f"Top 10 {geographic_level}s by Sales")
               fig.update_xaxes(tickangle=45)
               st.plotly_chart(fig, use_container_width=True)
           
           with col2:
               # Regional profit margins
               fig = px.bar(x=top_regions.index, y=top_regions['Profit Margin %'],
                          title=f"Profit Margins by {geographic_level}",
                          color=top_regions['Profit Margin %'],
                          color_continuous_scale='RdYlGn')
               fig.update_xaxes(tickangle=45)
               st.plotly_chart(fig, use_container_width=True)
           
           # Regional comparison matrix
           if len(regional_analysis) <= 10:  # Only for manageable number of regions
               st.markdown("#### 📊 Regional Performance Matrix")
               
               fig = px.scatter(regional_analysis, x='Total Sales', y='Profit Margin %',
                              size='Total Customers', hover_name=regional_analysis.index,
                              title="Regional Performance: Sales vs Profitability")
               st.plotly_chart(fig, use_container_width=True)
           
           # Growth opportunity analysis
           st.markdown("#### 🚀 Growth Opportunities")
           
           # Identify underperforming regions with potential
           median_sales_per_customer = regional_analysis['Sales per Customer'].median()
           median_orders_per_customer = regional_analysis['Orders per Customer'].median()
           
           growth_opportunities = regional_analysis[
               (regional_analysis['Sales per Customer'] < median_sales_per_customer) &
               (regional_analysis['Total Customers'] > regional_analysis['Total Customers'].quantile(0.3))
           ].head(5)
           
           if not growth_opportunities.empty:
               st.markdown("##### 🎯 Regions with Growth Potential")
               st.dataframe(growth_opportunities[['Total Sales', 'Total Customers', 'Sales per Customer', 'Market Share %']], 
                          use_container_width=True)
               
               st.info("💡 These regions have a good customer base but lower sales per customer, indicating potential for growth through targeted marketing and customer engagement strategies.")
           else:
               st.success("✅ All regions are performing well relative to their customer base.")
           
           # Underperforming regions needing attention
           underperforming = regional_analysis[
               (regional_analysis['Profit Margin %'] < 5) |
               (regional_analysis['Market Share %'] < 2)
           ].head(5)
           
           if not underperforming.empty:
               st.markdown("##### ⚠️ Regions Needing Attention")
               st.dataframe(underperforming[['Total Sales', 'Profit Margin %', 'Market Share %', 'Avg Delivery Days']], 
                          use_container_width=True)
               st.warning("⚠️ These regions show low profitability or market share and may need strategic intervention.")
           
           # Download regional report
           csv_data = regional_analysis.to_csv(index=True)
           st.download_button(
               label="📥 Download Regional Report",
               data=csv_data,
               file_name=f"regional_report_{geographic_level.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
               mime="text/csv"
           )
   
   # Executive Summary Report Generation
   st.markdown("---")
   st.subheader("📄 Executive Summary Generator")
   
   col1, col2 = st.columns([1, 2])
   
   with col1:
       st.markdown("#### Report Configuration")
       
       summary_type = st.selectbox(
           "Summary Type",
           ["Business Overview", "Financial Summary", "Performance Review", "Strategic Insights"]
       )
       
       include_charts = st.checkbox("Include Charts", value=True)
       include_recommendations = st.checkbox("Include Recommendations", value=True)
       
       if st.button("📊 Generate Executive Summary", type="primary"):
           generate_summary = True
       else:
           generate_summary = False
   
   with col2:
       if generate_summary:
           st.markdown(f"#### 📋 {summary_type}")
           
           # Generate executive summary based on type
           if summary_type == "Business Overview":
               st.markdown("##### 🏢 Business Performance Overview")
               
               # Key metrics
               total_revenue = filtered_df['Sales'].sum()
               total_profit = filtered_df['Profit'].sum()
               profit_margin = (total_profit / total_revenue) * 100
               total_orders = filtered_df['Order ID'].nunique()
               total_customers = filtered_df['Customer ID'].nunique()
               
               overview_text = f"""
               **Executive Summary - Business Overview**
               
               Our analysis of the SuperStore business reveals the following key insights:
               
               **Financial Performance:**
               • Total Revenue: ${total_revenue:,.0f}
               • Total Profit: ${total_profit:,.0f}
               • Overall Profit Margin: {profit_margin:.1f}%
               • Total Orders Processed: {total_orders:,}
               • Active Customer Base: {total_customers:,}
               
               **Market Position:**
               • Average Order Value: ${total_revenue/total_orders:.0f}
               • Customer Lifetime Value: ${total_revenue/total_customers:.0f}
               • Orders per Customer: {total_orders/total_customers:.1f}
               """
               
               st.markdown(overview_text)
               
               # Top performers
               top_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
               top_category = filtered_df.groupby('Product Category')['Sales'].sum().idxmax()
               top_segment = filtered_df.groupby('Customer Segment')['Sales'].sum().idxmax()
               
               st.markdown(f"""
               **Top Performers:**
               • Leading Region: **{top_region}**
               • Best Product Category: **{top_category}**
               • Most Valuable Segment: **{top_segment}**
               """)
           
           elif summary_type == "Financial Summary":
               st.markdown("##### 💰 Financial Performance Summary")
               
               # Financial metrics calculation
               total_revenue = filtered_df['Sales'].sum()
               total_profit = filtered_df['Profit'].sum()
               total_costs = total_revenue - total_profit
               profit_margin = (total_profit / total_revenue) * 100
               
               # Category-wise financials
               category_financials = filtered_df.groupby('Product Category').agg({
                   'Sales': 'sum',
                   'Profit': 'sum'
               })
               category_financials['Profit_Margin'] = (category_financials['Profit'] / category_financials['Sales'] * 100)
               
               best_category = category_financials['Profit_Margin'].idxmax()
               worst_category = category_financials['Profit_Margin'].idxmin()
               
               financial_summary = f"""
               **Financial Performance Analysis**
               
               **Overall Financial Health:**
               • Total Revenue: ${total_revenue:,.0f}
               • Total Profit: ${total_profit:,.0f}
               • Total Costs: ${total_costs:,.0f}
               • Profit Margin: {profit_margin:.1f}%
               • ROI: {(total_profit/total_costs)*100:.1f}%
               
               **Profitability Analysis:**
               • Highest Margin Category: **{best_category}** ({category_financials.loc[best_category, 'Profit_Margin']:.1f}%)
               • Lowest Margin Category: **{worst_category}** ({category_financials.loc[worst_category, 'Profit_Margin']:.1f}%)
               
               **Financial Recommendations:**
               • {"✅ Strong profitability" if profit_margin > 15 else "⚠️ Monitor profit margins closely"}
               • {"Focus on scaling high-margin products" if category_financials['Profit_Margin'].max() > 20 else "Review pricing strategy"}
               """
               
               st.markdown(financial_summary)
           
           elif summary_type == "Performance Review":
               st.markdown("##### 📈 Performance Review Summary")
               
               # Performance metrics
               current_period = filtered_df
               
               # Calculate growth (simplified approach)
               monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum()
               if len(monthly_sales) > 1:
                   recent_month = monthly_sales.iloc[-1]
                   previous_month = monthly_sales.iloc[-2]
                   monthly_growth = ((recent_month - previous_month) / previous_month * 100)
               else:
                   monthly_growth = 0
               
               # Performance indicators
               avg_delivery = filtered_df['Delivery_Days'].mean()
               customer_satisfaction = 85  # Placeholder
               order_fulfillment = 98.5  # Placeholder
               
               performance_review = f"""
               **Performance Review Summary**
               
               **Sales Performance:**
               • Monthly Growth Rate: {monthly_growth:+.1f}%
               • Revenue Trend: {"📈 Positive" if monthly_growth > 0 else "📉 Declining"}
               • Market Position: {"Strong" if monthly_growth > 5 else "Stable"}
               
               **Operational Performance:**
               • Average Delivery Time: {avg_delivery:.1f} days
               • Order Fulfillment Rate: {order_fulfillment:.1f}%
               • Customer Satisfaction: {customer_satisfaction:.1f}%
               
               **Key Performance Indicators:**
               • Revenue per Customer: ${filtered_df['Sales'].sum()/filtered_df['Customer ID'].nunique():.0f}
               • Average Order Processing: {avg_delivery:.1f} days
               • Customer Retention: {"High" if filtered_df.groupby('Customer ID')['Order ID'].nunique().mean() > 2 else "Moderate"}
               """
               
               st.markdown(performance_review)
           
           else:  # Strategic Insights
               st.markdown("##### 💡 Strategic Insights Summary")
               
               # Strategic analysis
               profit_margin = (filtered_df['Profit'].sum() / filtered_df['Sales'].sum()) * 100
               
               # Market concentration
               region_concentration = filtered_df.groupby('Region')['Sales'].sum()
               top_region_share = region_concentration.max() / region_concentration.sum() * 100
               
               # Customer analysis
               customer_segments = filtered_df.groupby('Customer Segment')['Sales'].sum()
               dominant_segment = customer_segments.idxmax()
               dominant_segment_share = customer_segments.max() / customer_segments.sum() * 100
               
               strategic_insights = f"""
               **Strategic Business Insights**
               
               **Market Position:**
               • Overall Profitability: {profit_margin:.1f}% {"(Healthy)" if profit_margin > 10 else "(Needs Improvement)"}
               • Market Concentration: {top_region_share:.1f}% in top region
               • Customer Base: {dominant_segment} segment dominates ({dominant_segment_share:.1f}%)
               
               **Strategic Opportunities:**
               • {"✅ Expand successful strategies to underperforming regions" if top_region_share > 40 else "Focus on balanced regional growth"}
               • {"Diversify customer segments" if dominant_segment_share > 50 else "Maintain balanced segment approach"}
               • {"Optimize product mix for higher margins" if profit_margin < 15 else "Scale current successful products"}
               
               **Risk Factors:**
               • {"High regional concentration risk" if top_region_share > 50 else "Balanced regional distribution"}
               • {"Customer segment concentration" if dominant_segment_share > 60 else "Diversified customer base"}
               • {"Margin pressure" if profit_margin < 10 else "Healthy profit margins"}
               
               **Strategic Recommendations:**
               1. {"Focus on margin improvement initiatives" if profit_margin < 15 else "Scale high-performing segments"}
               2. {"Diversify geographic presence" if top_region_share > 40 else "Optimize regional operations"}
               3. {"Develop customer retention programs" if filtered_df.groupby('Customer ID')['Order ID'].nunique().mean() < 2 else "Expand customer acquisition"}
               """
               
               st.markdown(strategic_insights)
           
           # Add charts if requested
           if include_charts:
               st.markdown("---")
               st.markdown("##### 📊 Supporting Charts")
               
               col1, col2 = st.columns(2)
               
               with col1:
                   # Revenue by category
                   cat_revenue = filtered_df.groupby('Product Category')['Sales'].sum()
                   fig = px.pie(values=cat_revenue.values, names=cat_revenue.index,
                              title="Revenue Distribution by Category")
                   st.plotly_chart(fig, use_container_width=True)
               
               with col2:
                   # Monthly trend
                   monthly_trend = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum()
                   fig = px.line(x=monthly_trend.index.astype(str), y=monthly_trend.values,
                               title="Monthly Revenue Trend")
                   st.plotly_chart(fig, use_container_width=True)
           
           # Add recommendations if requested
           if include_recommendations:
               st.markdown("---")
               st.markdown("##### 🎯 Executive Recommendations")
               
               recommendations = []
               
               # Profit-based recommendations
               if profit_margin < 10:
                   recommendations.append("🔴 **CRITICAL**: Implement immediate cost reduction and pricing optimization measures")
               elif profit_margin < 15:
                   recommendations.append("🟡 **MEDIUM**: Focus on improving operational efficiency and product mix")
               else:
                   recommendations.append("🟢 **GOOD**: Consider expansion opportunities and market growth strategies")
               
               # Regional recommendations
               if top_region_share > 50:
                   recommendations.append("🌍 **GEOGRAPHIC**: Diversify market presence to reduce regional dependency risk")
               
               # Customer recommendations
               repeat_customer_rate = len(filtered_df.groupby('Customer ID').filter(lambda x: len(x) > 1)) / filtered_df['Customer ID'].nunique()
               if repeat_customer_rate < 0.3:
                   recommendations.append("👥 **CUSTOMER**: Develop customer retention and loyalty programs")
               
               # Product recommendations
               loss_making_orders = len(filtered_df[filtered_df['Profit'] < 0])
               if loss_making_orders > len(filtered_df) * 0.1:
                   recommendations.append("📦 **PRODUCT**: Review and optimize underperforming product lines")
               
               for i, rec in enumerate(recommendations, 1):
                   st.markdown(f"{i}. {rec}")
           
           # Export options
           st.markdown("---")
           col1, col2, col3 = st.columns(3)
           
           with col1:
               if st.button("📄 Export as Text"):
                   # Create text version of summary
                   summary_text = f"""
EXECUTIVE SUMMARY - {summary_type.upper()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{overview_text if summary_type == "Business Overview" else 
  financial_summary if summary_type == "Financial Summary" else
  performance_review if summary_type == "Performance Review" else
  strategic_insights}

---
Report generated by SuperStore Analytics Dashboard
                   """
                   
                   st.download_button(
                       label="📥 Download Summary",
                       data=summary_text,
                       file_name=f"executive_summary_{summary_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                       mime="text/plain"
                   )
           
           with col2:
               if st.button("📊 Export Data"):
                   # Export underlying data
                   summary_data = filtered_df.groupby(['Region', 'Product Category', 'Customer Segment']).agg({
                       'Sales': 'sum',
                       'Profit': 'sum',
                       'Order ID': 'nunique',
                       'Customer ID': 'nunique'
                   }).reset_index()
                   
                   csv_data = summary_data.to_csv(index=False)
                   st.download_button(
                       label="📥 Download Data",
                       data=csv_data,
                       file_name=f"summary_data_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv"
                   )
           
           with col3:
               if st.button("🔄 Refresh Analysis"):
                   st.rerun()

# Final section - Data Export Center
st.markdown("---")
st.subheader("📤 Data Export Center")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Quick Exports")
    
    if st.button("📥 Export All Data", type="primary"):
        # Export filtered dataset
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset",
            data=csv_data,
            file_name=f"superstore_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    if st.button("📈 Export Summary Statistics"):
        # Create summary statistics
        summary_stats = filtered_df.describe()
        csv_data = summary_stats.to_csv()
        st.download_button(
            label="📥 Download Summary Stats",
            data=csv_data,
            file_name=f"summary_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    st.markdown("#### 📋 Custom Reports")
    
    # Custom report builder
    custom_groupby = st.selectbox(
        "Group By",
        ["Product Category", "Region", "Customer Segment", "Month", "State or Province"],
        key="custom_export"
    )
    
    custom_metrics = st.multiselect(
        "Include Metrics",
        ["Sales", "Profit", "Quantity", "Orders", "Customers", "Profit Margin"],
        default=["Sales", "Profit", "Orders"],
        key="custom_metrics"
    )
    
    if st.button("📊 Generate Custom Report"):
        # Generate custom report
        if custom_groupby == "Month":
            custom_data = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))
        else:
            custom_data = filtered_df.groupby(custom_groupby)
        
        custom_report = {}
        
        if "Sales" in custom_metrics:
            custom_report["Sales"] = custom_data['Sales'].sum()
        if "Profit" in custom_metrics:
            custom_report["Profit"] = custom_data['Profit'].sum()
        if "Quantity" in custom_metrics:
            custom_report["Quantity"] = custom_data['Quantity ordered new'].sum()
        if "Orders" in custom_metrics:
            custom_report["Orders"] = custom_data['Order ID'].nunique()
        if "Customers" in custom_metrics:
            custom_report["Customers"] = custom_data['Customer ID'].nunique()
        if "Profit Margin" in custom_metrics:
            custom_report["Profit Margin"] = (custom_data['Profit'].sum() / custom_data['Sales'].sum() * 100)
        
        custom_df = pd.DataFrame(custom_report).round(2)
        
        csv_data = custom_df.to_csv(index=True)
        st.download_button(
            label="📥 Download Custom Report",
            data=csv_data,
            file_name=f"custom_report_{custom_groupby.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    st.markdown("#### ℹ️ Export Information")
    
    st.info(f"""
    **Current Dataset:**
    • Records: {len(filtered_df):,}
    • Date Range: {filtered_df['Order Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Order Date'].max().strftime('%Y-%m-%d')}
    • Filters Applied: {"Yes" if len(filtered_df) < len(df) else "No"}
    """)
    
    st.success("""
    **Export Formats:**
    • CSV: Data tables
    • TXT: Text summaries
    • All exports include timestamp
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>
        <a href="https://twitter.com/yourhandle" target="_blank">Twitter</a> |
        <a href="https://linkedin.com/in/yourhandle" target="_blank">LinkedIn</a> |
        <a href="https://github.com/yourhandle" target="_blank">GitHub</a>
    </p>
    <p>&copy; Paul Anam 2025 <em>SuperStore Analytics Dashboard</em></p>
</div>
""", unsafe_allow_html=True)