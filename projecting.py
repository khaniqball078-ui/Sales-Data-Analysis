# ============================================================
# SALES DATA ANALYSIS PROJECT
# Author: Abdullah Khan
# Description: Exploratory Data Analysis using Pandas
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES AND LOAD DATASET
# ============================================================

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1206)
pd.set_option('display.max_rows', 10)

dash = pd.read_csv(r"C:\Users\Abdullah\Favorites\Downloads\sales_data_analysis.csv")


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n--- Dataset Preview ---")
print(dash.head())

print("\n--- Dataset Shape ---")
print(dash.shape)

print("\n--- Column Names ---")
print(dash.columns)

print("\n--- Data Types ---")
print(dash.dtypes)

print("\n--- Dataset Information ---")
dash.info()

print("\n--- Statistical Summary ---")
print(dash.describe())

print("\n--- Missing Values ---")
print(dash.isnull().sum())

print("\n--- Total Missing Values ---")
print(dash.isnull().sum().sum())

print("\n--- Duplicate Rows ---")
print(dash.duplicated().sum())

print("\n--- Unique Values in Each Column ---")
print(dash.nunique())

print("\n--- Unique Order Counts ---")
print(dash['order_id'].nunique())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Convert order_date into datetime format
dash['order_date'] = pd.to_datetime(
    dash['order_date'],
    errors='coerce'
)

# Remove complete duplicate rows
dash.drop_duplicates(inplace=True)

# Fill missing categorical values with NA
categorical_cols = dash.select_dtypes(
    include='object'
).columns

dash[categorical_cols] = dash[categorical_cols].fillna('NA')

print("\n--- Missing Values After Cleaning ---")
print(dash.isnull().sum())

print("\n--- Duplicate Rows After Cleaning ---")
print(dash.duplicated().sum())


# ============================================================
# 4. DATA QUALITY AND VALIDATION
# ============================================================

print("\n--- Quantity Less Than or Equal to Zero ---")
print(dash[dash['quantity'] <= 0])

print("\n--- Unit Price Less Than or Equal to Zero ---")
print(dash[dash['unit_price'] <= 0])

print("\n--- Zero Discount Transactions ---")
print(dash[dash['discount_pct'] == 0])

print("\n--- Negative Sales Transactions ---")
print(dash[dash['sales'] < 0])

# Calculate expected sales from quantity, price and discount
dash['calculated_sales'] = (
    dash['quantity']
    * dash['unit_price']
    * (1 - dash['discount_pct'] / 100)
)

# Compare calculated sales with recorded sales
dash['sales_difference'] = (
    dash['sales'] - dash['calculated_sales']
)

print("\n--- Sales Calculation Comparison ---")
print(
    dash[
        ['quantity', 'unit_price', 'discount_pct',
         'sales', 'calculated_sales', 'sales_difference']
    ].head(10)
)


# ============================================================
# 5. FILTERING AND SORTING
# ============================================================

print("\n--- Orders with Quantity Greater Than 3 ---")

filt = dash[dash['quantity'] > 3]
print(filt)


print("\n--- Transactions with Sales Above 2000 ---")

fi = dash[dash['sales'] > 2000]
print(fi)


print("\n--- Top 10 Highest Sales Transactions ---")

dash.sort_values(
    by='sales',
    ascending=False,
    inplace=True
)

print(dash.head(10))


print("\n--- Orders Sorted by Date ---")

dash.sort_values(
    by='order_date',
    ascending=True,
    inplace=True
)

print(dash.head(10))


# ============================================================
# 6. OVERALL SALES PERFORMANCE
# ============================================================

print("\n--- Overall Sales Statistics ---")

print(f"Total Revenue: {dash['sales'].sum():,.2f}")

print(f"Total Number of Records: {len(dash)}")

print(f"Total Unique Orders: {dash['order_id'].nunique()}")

print(f"Average Order Value: {dash['sales'].mean():,.2f}")

print(f"Average Unit Price: {dash['unit_price'].mean():,.2f}")

print(f"Highest Sales Transaction: {dash['sales'].max():,.2f}")

print(f"Lowest Sales Transaction: {dash['sales'].min():,.2f}")

print(f"Median Transaction Value: {dash['sales'].median():,.2f}")

print(f"Sales Standard Deviation: {dash['sales'].std():,.2f}")

print(f"Total Discount Percentage Points: {dash['discount_pct'].sum():,.2f}")


# Sales categories
dash['sales_category'] = np.select(
    [
        dash['sales'] < 1000,
        dash['sales'] <= 4000
    ],
    [
        'Low',
        'Medium'
    ],
    default='High'
)

print("\n--- Sales Categories ---")

print(
    dash[['sales', 'sales_category']].head(10)
)

print("\n--- Sales Category Distribution ---")

print(dash['sales_category'].value_counts())


# ============================================================
# 7. PRODUCT-WISE ANALYSIS
# ============================================================

print("\n--- Top 5 Products by Quantity Sold ---")

tops = (
    dash.groupby('product')['quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name='Total_Quantity')
)

print(tops)


print("\n--- Top 5 Products by Revenue ---")

top = (
    dash.groupby('product')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name='Total_Revenue')
)

print(top)


print("\n--- Bottom 5 Products by Revenue ---")

topp = (
    dash.groupby('product')['sales']
    .sum()
    .sort_values(ascending=True)
    .head(5)
    .reset_index(name='Total_Revenue')
)

print(topp)


print("\n--- Average Sales by Product ---")

ave = (
    dash.groupby('product')['sales']
    .mean()
    .sort_values(ascending=False)
    .reset_index(name='Average_Sales')
)

print(ave)


print("\n--- Product-wise Revenue Contribution ---")

sales_total = dash['sales'].sum()

percet = dash.groupby('product')['sales'].sum()

contribution = (
    (percet / sales_total * 100)
    .sort_values(ascending=False)
    .reset_index(name='Revenue_Contribution_Percentage')
)

print(contribution)


# ============================================================
# 8. CATEGORY-WISE ANALYSIS
# ============================================================

print("\n--- Total Quantity Sold by Category ---")

total = (
    dash.groupby('category')['quantity']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Quantity')
)

print(total)


print("\n--- Revenue by Category ---")

reve = (
    dash.groupby('category')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(reve)


# ============================================================
# 9. REGION AND CITY ANALYSIS
# ============================================================

print("\n--- Revenue by Region ---")

reg = (
    dash.groupby('region')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(reg)


print("\n--- Revenue by City ---")

city = (
    dash.groupby('city')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(city)

print("\n--- Top 5 Cities by Revenue ---")
print(city.head(5))


print("\n--- Regional Revenue Contribution ---")

regs = dash['sales'].sum()

percets = dash.groupby('region')['sales'].sum()

reven = (
    (percets / regs * 100)
    .sort_values(ascending=False)
    .reset_index(name='Revenue_Contribution_Percentage')
)

print(reven)


print("\n--- Number of Orders by Region ---")

ords = (
    dash.groupby('region')['order_id']
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name='Total_Orders')
)

print(ords)


# ============================================================
# 10. DATE AND MONTH-WISE SALES ANALYSIS
# ============================================================

# Extract date components
dash['year'] = dash['order_date'].dt.year
dash['month'] = dash['order_date'].dt.month
dash['month_name'] = dash['order_date'].dt.month_name()
dash['date'] = dash['order_date'].dt.date

print("\n--- Extracted Date Information ---")

print(
    dash[
        ['order_date', 'year', 'month', 'month_name', 'date']
    ].head()
)


print("\n--- Monthly Revenue ---")

mon = (
    dash.groupby(['year', 'month', 'month_name'])['sales']
    .sum()
    .reset_index(name='Total_Revenue')
    .sort_values(['year', 'month'])
)

print(mon)


print("\n--- Highest Revenue Month ---")
print(mon.loc[mon['Total_Revenue'].idxmax()])


print("\n--- Lowest Revenue Month ---")
print(mon.loc[mon['Total_Revenue'].idxmin()])


print("\n--- Monthly Number of Orders ---")

orde = (
    dash.groupby(['year', 'month', 'month_name'])['order_id']
    .nunique()
    .reset_index(name='Total_Orders')
    .sort_values(['year', 'month'])
)

orde['Monthly_Order_Growth_Percentage'] = (
    orde.groupby('year')['Total_Orders']
    .pct_change() * 100
)

print(orde)


# ============================================================
# 11. CUSTOMER ANALYSIS
# ============================================================

print("\n--- Total Unique Customers ---")

cust = dash['customer_id'].nunique()
print(cust)


print("\n--- Top 10 Customers by Spending ---")

sped = (
    dash.groupby('customer_id')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name='Total_Spending')
)

print(sped)


print("\n--- Number of Orders by Customer ---")

orders_number = (
    dash.groupby('customer_id')['order_id']
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name='Total_Orders')
)

print(orders_number)


print("\n--- Average Revenue per Customer ---")

aves = dash['sales'].sum()
tots = dash['customer_id'].nunique()

print(f"Average Spending per Customer: {aves / tots:,.2f}")


print("\n--- Total Revenue by Customer ---")

total_revenue = (
    dash.groupby('customer_id')['sales']
    .sum()
    .reset_index(name='Total_Revenue')
)

print(total_revenue)


# ============================================================
# 12. ORDER STATUS ANALYSIS
# ============================================================

print("\n--- Number of Orders by Status ---")

stat = (
    dash.groupby('order_status')['order_id']
    .nunique()
    .sort_values(ascending=False)
)

print(stat)


print("\n--- Revenue by Order Status ---")

sale = (
    dash.groupby('order_status')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(sale)


print("\n--- Order Status Distribution (%) ---")

ors = (
    dash['order_status']
    .value_counts(normalize=True) * 100
)

print(ors)


# ============================================================
# 13. PAYMENT METHOD ANALYSIS
# ============================================================

print("\n--- Number of Transactions by Payment Method ---")

pay = (
    dash.groupby('payment_method')['order_id']
    .count()
    .sort_values(ascending=False)
    .reset_index(name='Total_Transactions')
)

print(pay)


print("\n--- Revenue by Payment Method ---")

pays = (
    dash.groupby('payment_method')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(pays)


# ============================================================
# 14. SALES CHANNEL ANALYSIS
# ============================================================

print("\n--- Revenue by Sales Channel ---")

channel = (
    dash.groupby('sales_channel')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index(name='Total_Revenue')
)

print(channel)


print("\n--- Sales Channel Distribution (%) ---")

pa = (
    dash['sales_channel']
    .value_counts(normalize=True) * 100
)

print(pa.sort_values(ascending=False))


# ============================================================
# 15. FINAL DATASET CHECK
# ============================================================

print("\n--- Final Dataset Shape ---")
print(dash.shape)

print("\n--- Final Dataset Preview ---")
print(dash.head())

print("\n--- Remaining Missing Values ---")
print(dash.isnull().sum())

print("\n--- Final Duplicate Count ---")
print(dash.duplicated().sum())


# ============================================================
# END OF PROJECT
# ============================================================


