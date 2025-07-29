import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('../datasets/SuperStoreUS.csv')

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Title
st.title("📊 SuperStore US - Sales & Profit Analysis")
st.subheader("Exercise 1: Tasks 4 & 5")

# Task 4 - Outlier Detection
st.header("📌 Task 4: Outlier Detection and Handling")

# Functions
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores > threshold]

# Sales Analysis
st.subheader("📈 Sales Outliers")
st.write(f"**Mean**: ${df['Sales'].mean():.2f} | **Median**: ${df['Sales'].median():.2f} | **Std Dev**: ${df['Sales'].std():.2f}")

sales_outliers_iqr, sales_lower, sales_upper = detect_outliers_iqr(df, 'Sales')
st.write(f"**IQR Bounds**: ${sales_lower:.2f} – ${sales_upper:.2f} | Outliers: {len(sales_outliers_iqr)} ({len(sales_outliers_iqr)/len(df)*100:.1f}%)")

sales_outliers_zscore = detect_outliers_zscore(df, 'Sales')
st.write(f"**Z-Score Outliers (Z > 3)**: {len(sales_outliers_zscore)} ({len(sales_outliers_zscore)/len(df)*100:.1f}%)")

# Profit Analysis
st.subheader("💰 Profit Outliers")
st.write(f"**Mean**: ${df['Profit'].mean():.2f} | **Median**: ${df['Profit'].median():.2f} | **Std Dev**: ${df['Profit'].std():.2f}")

profit_outliers_iqr, profit_lower, profit_upper = detect_outliers_iqr(df, 'Profit')
st.write(f"**IQR Bounds**: ${profit_lower:.2f} – ${profit_upper:.2f} | Outliers: {len(profit_outliers_iqr)} ({len(profit_outliers_iqr)/len(df)*100:.1f}%)")

profit_outliers_zscore = detect_outliers_zscore(df, 'Profit')
st.write(f"**Z-Score Outliers (Z > 3)**: {len(profit_outliers_zscore)} ({len(profit_outliers_zscore)/len(df)*100:.1f}%)")

# Visualizations
st.subheader("📉 Outlier Visualizations")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Sales & Profit Distributions', fontsize=16)

axes[0, 0].boxplot(df['Sales'])
axes[0, 0].set_title('Sales - Boxplot')

axes[0, 1].hist(df['Sales'], bins=50, color='skyblue')
axes[0, 1].set_title('Sales - Histogram')

axes[1, 0].boxplot(df['Profit'])
axes[1, 0].set_title('Profit - Boxplot')

axes[1, 1].hist(df['Profit'], bins=50, color='lightgreen')
axes[1, 1].set_title('Profit - Histogram')

st.pyplot(fig)
# Task 5 - Data Quality Report
st.header("📋 Task 5: Data Quality Report")

def create_data_quality_report(df):
    st.subheader("🔍 Data Quality Assessment Report")
    st.markdown(f"Generated: `{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}`")

    # Dataset Overview
    st.markdown("### 1. Dataset Overview")
    st.write(f"- Total Rows: **{len(df):,}**")
    st.write(f"- Total Columns: **{len(df.columns)}**")
    st.write(f"- Memory Usage: **{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB**")

    # Missing Values
    st.markdown("### 2. Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ No missing values found.")
    else:
        st.error("❌ Missing values detected:")
        st.dataframe(missing_data[missing_data > 0])

    # Data Types
    st.markdown("### 3. Data Types Analysis")
    st.write(df.dtypes.value_counts())

    # Duplicates
    st.markdown("### 4. Duplicate Records")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.error(f"❌ {dup_count} duplicate rows found ({dup_count/len(df)*100:.1f}%)")

    # Numerical Columns
    st.markdown("### 5. Numerical Columns Quality")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        st.markdown(f"**• {col}**")
        st.write(f"- Range: ${df[col].min():.2f} to ${df[col].max():.2f}")
        st.write(f"- Zero Values: {(df[col]==0).sum()} | Negative Values: {(df[col]<0).sum()}")
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
        st.write(f"- Outliers (IQR): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

    # Categorical Columns
    st.markdown("### 6. Categorical Columns Quality")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if col not in ['Order Date', 'Ship Date']:
            st.markdown(f"**• {col}**")
            st.write(f"- Unique: {df[col].nunique()} | Most common: {df[col].mode()[0]}")

    # Dates
    st.markdown("### 7. Date Columns Quality")
    for date_col in ['Order Date', 'Ship Date']:
        if date_col in df.columns:
            st.markdown(f"**• {date_col}**")
            st.write(f"- Range: {df[date_col].min().date()} → {df[date_col].max().date()}")
            st.write(f"- Nulls: {df[date_col].isnull().sum()}")

    # Business Logic
    st.markdown("### 8. Business Logic Validation")
    invalid_ship_dates = df[df['Ship Date'] < df['Order Date']]
    if len(invalid_ship_dates) == 0:
        st.success("✅ All ship dates are after order dates.")
    else:
        st.error(f"❌ {len(invalid_ship_dates)} rows where ship date < order date.")

    high_profit_low_sales = df[(df['Profit'] > df['Sales']) & (df['Sales'] > 0)]
    if len(high_profit_low_sales) == 0:
        st.success("✅ Profit does not exceed Sales.")
    else:
        st.warning(f"⚠️ {len(high_profit_low_sales)} cases where Profit > Sales.")

    # Score
    st.markdown("### 9. Data Quality Score")
    score = 100
    if missing_data.sum() > 0:
        score -= 10
    if dup_count > 0:
        score -= 5
    if len(invalid_ship_dates) > 0:
        score -= 10

    st.metric("Overall Quality Score", f"{score}/100")
    if score >= 90:
        st.success("🟢 Excellent quality.")
    elif score >= 75:
        st.info("🟡 Good quality.")
    elif score >= 60:
        st.warning("🟠 Fair quality.")
    else:
        st.error("🔴 Poor quality.")

    # Recommendations
    st.markdown("### 10. Recommendations")
    recs = []
    if missing_data.sum() > 0:
        recs.append("• Impute or drop missing values.")
    if dup_count > 0:
        recs.append("• Remove duplicate records.")
    if len(invalid_ship_dates) > 0:
        recs.append("• Investigate ship date inconsistencies.")
    if not recs:
        recs.append("• No major issues detected.")
    for rec in recs:
        st.write(rec)

# Display report
create_data_quality_report(df)

# Summary Stats
st.subheader("📊 Summary Statistics Table")
cols = ['Sales', 'Profit', 'Discount', 'Quantity ordered new', 'Shipping Cost']
existing_cols = [c for c in cols if c in df.columns]
st.dataframe(df[existing_cols].describe().T.round(2))

st.success("✅ Exercise 1 Completed: Tasks 4 and 5 Done")
