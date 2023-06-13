# -*- coding: utf-8 -*-
"""
Author   : Qian Dong
Date     : 6/6/2023
Document : EBRD IAF assignment_code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


#%% 
'''
1. Go to https://www.beeps-ebrd.com/ and download the BEEPS 2018-2020 dataset 
in suitable format. You will need to register in order to access the data. 
Familiarise yourself with the questionnaire and documentation available.
'''
# Read data and observe data shape
df = pd.read_csv('beeps_vi_es_csv.csv', encoding='latin1') 
rr, cc = df.shape

#%% 
'''
2. Create a new variable combining variables a2x and country. This variable will 
display the region and the country together and should look like this: Tbilisi, 
Georgia.
'''
# Create new column 'region_country'
df['region_country'] = df['a2x'] + ', ' + df['country']
print(df['region_country'].head()) # Show top 5 in new column

#%% 
'''
3. For all observations in the dataset, create a variable with the percentage 
of female top managers in the country. Find countries with the highest and 
lowest percentage of female top managers.
'''
# Column b7a: Is The Top Manager Female? 
# -9 Don't know (spontaneous) 
# 1 Yes 
# 2 No 

# Extract column information 'country' and 'b7a'
countries = df['country'].unique() # All countries listed, unrepeating
country_count = df['country'].value_counts() # Number of companies in each country

# Create dictionary of country: female top percentage
female_top_percentage = {}
for i in countries:
    female_top_percentage[i] = 0

# Count the number of female tops in each country by looping through dataset
for r in range(rr):
    if df['b7a'].iloc[r] == 1:
        female_top_percentage[df['country'].iloc[r]] += 1

# Divide dictionary values by total number of companies to get percentage values
for c in female_top_percentage.keys():
    female_top_percentage[c] = female_top_percentage[c] / country_count[c] * 100

# Create a new column of country percentage value, by mapping with the dictionary
df['female_top_percentage'] = df['country'].map(female_top_percentage)
print(df['female_top_percentage'].head()) # Show top 5 in new column

# Find and display highest/lowest percentage countries
highest = list(female_top_percentage.keys())[list(female_top_percentage.values()).index(max(list(female_top_percentage.values())))]
lowest = list(female_top_percentage.keys())[list(female_top_percentage.values()).index(min(list(female_top_percentage.values())))]

print("Country with the highest percentage of female top managers: " + highest)
print("Country with the lowest percentage of female top managers: " + lowest)

#%% 
'''
4. Variable m1a contains the most important obstacle as selected by the 
respondent. For each country, we want to know the percentage of respondents that 
selected each obstacle (what percentage of respondents in country A selected 
obstacle 1, what percentage in country A selected obstacle 2, etc., up to 
obstacle 15). Present the results in a table format.
'''
# Column M1A: Biggest Obstacle Affecting The Operation of This Establishment 
#  -9 Don't know (spontaneous) 
#  -7 Does not apply 
#  1 Access to finance 
#  2 Access to land 
#  3 Business licensing and permits 
#  4 Corruption 
#  5 Courts 
#  6 Crime, theft and disorder 
#  7 Customs and trade regulations 
#  8 Electricity 
#  9 Inadequately educated workforce 
#  10 Labor regulations 
#  11 Political instability 
#  12 Practices of competitors in the informal sector 
#  13 Tax administration 
#  14 Tax rates 
#  15 Transport 

# Create a pivot table counting number of selections in each country
temp = pd.pivot_table(df, index = 'country', columns = 'm1a', aggfunc = 'size', fill_value = 0)

# Convert to percentages
obstacle_percentage = temp.apply(lambda x: x / x.sum() * 100, axis = 1)
print(obstacle_percentage.head()) # Show top 5 rows of new dataframe

#%% 
'''
5. Represent the results from point 4 for Turkey only in a bar chart.
'''
# Extract the row for Turkey from dataframe
p_Turkey = obstacle_percentage.loc['Turkey']

# Plot bar chart
plt.figure(figsize = (15, 8))
plt.bar(p_Turkey.index, p_Turkey)
plt.title("Percentage of Most Important Obstacles, Turkey")
plt.xlabel("Obstacle Index")
plt.ylabel("Percentage")
plt.show()

#%% 
'''
6. Do the same task as in point 4 for Turkey only (only table), but this time 
by size of firm (i.e., number of employees). Use the following size classes: 
1-19, 20-99 and 100+. Hint: There is a variable that already defines size 
classes in the dataset. 
'''
# Column stratificationsizecode: Size of firm
#  1 Small (5-19) 
#  2 Medium (20-99) 
#  3 Large (100 or more)

# Create a new copy of dataframe with Turkey data only
df_Turkey = df.loc[df['country'] == 'Turkey'].copy()

df['stratificationsizecode'].unique() # Column contains unique values [2, 3, 1, 4]
df_Turkey['stratificationsizecode'] = df_Turkey['stratificationsizecode'].replace(4, 3)

# Create a pivot table counting number of selections in each size division
temp1 = pd.pivot_table(df_Turkey, index = 'stratificationsizecode', columns = 'm1a', aggfunc = 'size', fill_value = 0)

# Convert to percentages
obstacle_percentage_Turkey = temp1.apply(lambda x: x / x.sum() * 100, axis = 1)
print(obstacle_percentage_Turkey.head()) # Show top 5 rows of new dataframe

#%% 
'''
7. Go to the World Development Indicators database 
(https://databank.worldbank.org/source/world-development-indicators) and 
download the gross domestic product (GDP) for 2019, in current prices (in US 
dollars), for each of the countries in the dataset. Then combine the GDP data 
with the BEEPS dataset.
'''
# File name: GDP_Data.csv
# Read  GDPdata and modify column names to enable merging
GDPdata = pd.read_csv('GDP_Data.csv') 
GDPdata.rename(columns = {'Country Name': 'country', '2019 [YR2019]': 'GDP2019'}, inplace=True)
print(GDPdata.head()) # Show top 5 rows of new dataframe

# Merge df and GDPdata by country
df_GDP = pd.merge(df, GDPdata, on='country')
print(df_GDP.head()) # Show top 5 rows of new dataframe

#%% 
'''
8. Using the GDP data from point 7, create a scatter plot comparing GDP and 
average capacity utilization (f1) in the country for all countries in one graph, 
i.e. there should only be one point per country. 
'''
# Remove -9 and empty values from column f1
# Calculate average f1 value of each country
f1_avr = df.loc[(df['f1'] >= 0) & (df['f1'] != np.nan)].groupby('country')['f1'].mean()

# Merge average f1 with the GDP
f1_GDP = pd.merge(f1_avr, GDPdata[['country', 'GDP2019']], on='country')

# Initialise plot
plt.figure(figsize = (15, 8))
plt.scatter(f1_GDP['GDP2019'], f1_GDP['f1'])

# Add country labels 
for i, txt in enumerate(f1_GDP['country']):
    plt.annotate(txt, (f1_GDP['GDP2019'][i], f1_GDP['f1'][i]))

# Modify and show plot
plt.xlabel('GDP,2019')
plt.ylabel('average capacity utilization (f1)')
plt.title('Scatter plot of average capacity utilization (f1) to GDP, 2019')
plt.show()

#%% 
'''
9. Run a regression with sales (d2) as a left-hand-side (LHS) variable and access 
to finance as an obstacle (k30), plus any controls that you think should be 
included on the right-hand-side (RHS). Interpret the coefficient for the access 
to finance as an obstacle. If there are any problems with the estimation, write 
what you think they are.
'''
# Column K30: How Much of An Obstacle: Access To Finance 
#  -9 Don't know (spontaneous) 
#  -7 Does not apply 
#  0 No obstacle 
#  1 Minor obstacle 
#  2 Moderate obstacle 
#  3 Major obstacle 
#  4 Very severe obstacle 

# Filter out all invalid values in targeted data columns from df
df_filtered = df_GDP.loc[(df_GDP['d2'] >= 0) # sales
                     & (df_GDP['k30'] >= 0) # Access To Finance 
                     # Additional controls:
                     & (df_GDP['e1'] >= 0) # In Last FY, Main Market For Establishment'S Main Product 
                     & (df_GDP['GDP2019'] >= 0) # Country GDP in 2019
                     ]

# Name the 2 main variables
sales = df_filtered['d2']
finance_obst = df_filtered['k30']

# Identify RHS, LHS variables, run regression
X = sm.add_constant(np.column_stack((finance_obst, 
                                     df_filtered['e1'], 
                                     df_filtered['GDP2019'], 
                                     ))) # Add intercept to x 
y = sales
model = sm.OLS(y, X, missing = 'drop')
results = model.fit()

# print(results.summary())

# Inteprete results
print('sales | No obstacle in Access to Finance', 
      round(results.params.iloc[0]  , 4), '\n', 
      
      'coefficient: finance as an obstacle (k30)',  '\n', 
      round(results.params.iloc[1], 4), '\n', 
      'p-val',  '\n', 
      round(results.pvalues.iloc[1], 4), '\n', 
      
      'coefficient: In Last FY, Main Market For Establishments Main Product',  '\n', 
      round(results.params.iloc[1], 4), '\n', 
      'p-val',  '\n', 
      round(results.pvalues.iloc[2], 4), '\n', 
      
      'coefficient: Country GDP in 2019',  '\n', 
      round(results.params.iloc[3], 4), '\n', 
      'p-val',  '\n', 
      round(results.pvalues.iloc[3], 4)
      )

print('''
      As shown in the regression results, access to finance as an obstacle 
      negatively affects sales. As the level of obstacle increases by 1, sales 
      drop by around 1.25 billion. However, this result is not statistically 
      significant with p-value > 0.05.
      
      Other control factors have been considered through multiple 
      trial-and-errors, such as Main Market For Establishments Main Product in 
      last FY (p < 0.05), and Country GDP in 2019 (p > 0.05). Only Main Market 
      For Establishments Main Product is a valid control factor. 
      
      The main difficulties in finding valid controls are:
          1. Some variables have limited data, so taking them into account will 
          excessively shrink the size of regression data;
          2. Many variables (e.g. 'xx as an obstacle' series) are far from 
          significant controls;
          3. Some variables are recorded in string format, inconvenient for 
          regression running. 
      ''')
