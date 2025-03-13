# Internship-Program-Business-Analysis.

## Project Title: Customer Segmentation Visualization & Advanced Analysis

#### Project Overview: 
The project aims to analyze customer churn in a telecommunications company and develop predictive models to identify at-risk customers. The ultimate goal is to provide actionable insights and recommendations to reduce churn and improve customer retention.

```python
# Task - 1

# pip install pandas
import pandas as pd

# load the dataset
df = pd.read_csv(r"C:\Users\sanka\Downloads\Telco_Customer_Churn_Dataset  (3).csv")

# show data
df.info()

# display frist 10 rows 
df.head(10)

#show data types fro each column 
print(df.dtypes)
print(df)

# check null values 
df.isna().sum()    # there is no null values

# task-2

# there is ant null values drop na 
df.dropna(inplace = True)

# Find duplicate rows 
duplicates = df[df.duplicated()]
print(duplicates)

# Standardize column names 
df.columns = df.columns.str.lower()

#renames columns 
df = df.rename(columns = {'customerid' : 'customer_id'})
df = df.rename(columns = {'seniorcitizen' : 'senior_citizen'})
df = df.rename(columns = {'phoneservice' : 'phone_service'})
df = df.rename(columns = {'multiplelines' : 'multiple_lines'})
df = df.rename(columns = {'internetservice' : 'internet_service'})
df = df.rename(columns = {'onlinesecurity' : 'online_security'})
df = df.rename(columns = {'onlinebackup' : 'online_backup'})
df = df.rename(columns = {'deviceprotection' : 'device_protection'})
df = df.rename(columns = {'techsupport' : 'tech_support'})
df = df.rename(columns = {'streamingtv' : 'streaming_tv'})
df = df.rename(columns = {'streamingmovies' : 'streaming_movies'})
df = df.rename(columns = {'paperlessbilling' : 'paperless_billing'})
df = df.rename(columns = {'paymentmethod' : 'payment_method'})
df = df.rename(columns = {'monthlycharges' : 'monthly_charges'})
df = df.rename(columns = {'totalcharges' : 'total_charges'})

# change data types
# Remove rows with empty strings or non-numeric values
df = df[df['total_charges'].str.strip() != '']

# Convert to numeric
df['total_charges'] = pd.to_numeric(df['total_charges'])

# show column names 
df.info()
# replce space to "_"
df.columns  = df.columns.str.replace(" ","_")

# task -3 

# data distribution

# third moment business decisson (skewness)
# skewness value "0" data are normal skewed
# skewness value ">0" data are positively skewed
# skewness value "<0" data are negatively skewed

tenure_skew = df.tenure.skew()
print(tenure_skew)

monthly_charges_skew = df.monthly_charges.skew()
print(monthly_charges_skew)

total_charges_skew = df.total_charges.skew()
print(total_charges_skew)

# forth momment business decision (kurt)
# kurtosis value "3" data are normal 
# skewness value ">3" data are positive
# skewness value "<3" data are negative

tenure_kurt = df.tenure.kurt()
print(tenure_kurt)

monthly_charges_kurt = df.monthly_charges.kurt()
print(monthly_charges_skew)

total_charges_kurt = df.total_charges.kurt()
print(monthly_charges_kurt)

# finding mean 

senior_citizen_mean = df.senior_citizen.mean() 
print(senior_citizen_mean)

tenure_mean = df.tenure.mean()
print(tenure_mean)

monthly_charges_mean = df.monthly_charges.mean()
print(monthly_charges_mean)

total_charges_mean = df.total_charges.mean()
print(total_charges_mean)

#finding medain

senior_citizen_median = df.senior_citizen.median() 
print(senior_citizen_median)

tenure_median = df.tenure.median()
print(tenure_median)

monthly_charges_median = df.monthly_charges.median()
print(monthly_charges_median)

total_charges_median = df.total_charges.median()
print(total_charges_median)

#fine mode 

internet_service_mode = df.internet_service.mode()
print(internet_service_mode)

payment_method_mode = df.payment_method.mode()
print(payment_method_mode)

import matplotlib.pyplot as plt
import seaborn as sns

# column tenure_median :(histogram)
plt.hist(df.tenure)


# column monthly_charges :(histogram)
plt.hist(df.monthly_charges)

# column total_charges :(histogram)
plt.hist(df.total_charges)



# Create tenure categories
bins = [0, 12, 36, float('inf')]
labels = ['0-12 months', '13-36 months', '37+ months']
df['tenure_category'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)


# Create box plot for 'monthly_charges'
plt.figure(figsize=(8, 6))
sns.boxplot(x='tenure_category', y='monthly_charges', data=df, palette="Set2")
plt.title('Box Plot of Monthly Charges by Tenure Category')
plt.xlabel('Tenure Category')
plt.ylabel('Monthly Charges')
plt.show()

# Create box plot for 'total_charges'
plt.figure(figsize=(8, 6))
sns.boxplot(x='tenure_category', y='total_charges', data=df, palette="Set2")
plt.title('Box Plot of Total Charges by Tenure Category')
plt.xlabel('Tenure Category')
plt.ylabel('Total Charges')
plt.show()

# If you want to visualize these by other categories such as gender:
# Box plot for 'monthly_charges' by 'gender'
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='monthly_charges', data=df, palette="Set2")
plt.title('Box Plot of Monthly Charges by Gender')
plt.xlabel('Gender')
plt.ylabel('Monthly Charges')
plt.show()

# Box plot for 'total_charges' by 'gender'
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='total_charges', data=df, palette="Set2")
plt.title('Box Plot of Total Charges by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Charges')
plt.show()


#task - 4


# Step 1: Pie Chart (Customer Distribution by Tenure Category)
tenure_dist = df['tenure_category'].value_counts()

# Plotting the Donut chart
plt.figure(figsize=(8, 6))
plt.pie(tenure_dist, labels=tenure_dist.index, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3})
plt.title('Customer Distribution by Tenure')
plt.show()


#  Clustered Bar Chart for Average Monthly Charges
# Calculate the average monthly charges for each tenure category
avg_charges = df.groupby('tenure_category')['monthly_charges'].mean().reset_index()

# Plotting the clustered bar chart
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_charges, x='tenure_category', y='monthly_charges', palette='viridis')

# Adding annotations to highlight significant trends
for i, row in avg_charges.iterrows():
    plt.text(i, row['monthly_charges'] + 2, f'{row["monthly_charges"]:.2f}', 
             ha='center', va='bottom', fontsize=12, color='black')

plt.title('Average Monthly Charges Across Tenure Categories')
plt.xlabel('Tenure Category')
plt.ylabel('Average Monthly Charges')
plt.show()

# Convert 'churn' column to numeric (1 for churned, 0 for not churned)
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# Task - 5


# 1. Group by Tenure and Compute Statistics for Charges and Churn
tenure_grouped = df.groupby('tenure_category').agg(
    avg_monthly_charges=('monthly_charges', 'mean'),
    avg_total_charges=('total_charges', 'mean'),
    churn_rate=('churn', 'mean'),
    count=('churn', 'count')
).reset_index()

# Visualize the churn rate vs tenure (Churn vs Tenure trend)
plt.figure(figsize=(8, 6))
sns.lineplot(data=tenure_grouped, x='tenure_category', y='churn_rate', marker='o')
plt.title('Churn Rate by Tenure Category')
plt.xlabel('Tenure Category')
plt.ylabel('Churn Rate')
plt.show()

# 2. Churn Rate Analysis by Demographics (Gender, Senior Citizen Status)
# Churn rate by gender
churn_gender = df.groupby('gender').agg(churn_rate=('churn', 'mean')).reset_index()

# Churn rate by senior citizen status
churn_senior_citizen = df.groupby('senior_citizen').agg(churn_rate=('churn', 'mean')).reset_index()

# Churn rate by partner status
churn_partner = df.groupby('partner').agg(churn_rate=('churn', 'mean')).reset_index()

# Churn rate by dependents status
churn_dependents = df.groupby('dependents').agg(churn_rate=('churn', 'mean')).reset_index()

# Visualizing churn by gender
plt.figure(figsize=(8, 6))
sns.barplot(data=churn_gender, x='gender', y='churn_rate', palette='Set2')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()

# Visualizing churn by senior citizen status
plt.figure(figsize=(8, 6))
sns.barplot(data=churn_senior_citizen, x='senior_citizen', y='churn_rate', palette='Set2')
plt.title('Churn Rate by Senior Citizen Status')
plt.xlabel('Senior Citizen (0 = No, 1 = Yes)')
plt.ylabel('Churn Rate')
plt.show()

# Visualizing churn by partner status
plt.figure(figsize=(8, 6))
sns.barplot(data=churn_partner, x='partner', y='churn_rate', palette='Set2')
plt.title('Churn Rate by Partner Status')
plt.xlabel('Partner (No/Yes)')
plt.ylabel('Churn Rate')
plt.show()



# Visualizing churn by dependents status
plt.figure(figsize=(8, 6))
sns.barplot(data=churn_dependents, x='dependents', y='churn_rate', palette='Set2')
plt.title('Churn Rate by Dependents Status')
plt.xlabel('Dependents (No/Yes)')
plt.ylabel('Churn Rate')
plt.show()

# 3. Churn Rate by Payment Methods and Contract Types
churn_payment_method = df.groupby('payment_method').agg(churn_rate=('churn', 'mean')).reset_index()
churn_contract_type = df.groupby('contract').agg(churn_rate=('churn', 'mean')).reset_index()

# Churn rate by payment method
plt.figure(figsize=(10, 6))
sns.barplot(data=churn_payment_method, x='payment_method', y='churn_rate', palette='coolwarm')
plt.title('Churn Rate by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45, ha='right')
plt.show()

# Churn rate by contract type
plt.figure(figsize=(10, 6))
sns.barplot(data=churn_contract_type, x='contract', y='churn_rate', palette='coolwarm')
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45, ha='right')
plt.show()
```
