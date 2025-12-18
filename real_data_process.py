import pandas as pd

# Load the dataset
df = pd.read_excel('real_data/Intervention_data_spreadsheet_final.xlsx')
df = df.iloc[:, :6]
# Convert Start Date and End Date columns to datetime
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])

# Calculate the date 14 days after the specific date
specific_date = '2020-04-15'  # Specify the date of interest
date_plus_30 = pd.to_datetime(specific_date) + pd.Timedelta(days=30)

# Filter rows where the specific date falls between the Start Date and at least 14 days after the specific date in
# End Date Excluding 'mask mandate' policy
filtered_df = df[
    (df['Start Date'] <= specific_date) & (df['End Date'] >= date_plus_30) & (df['NPI measure'] != 'mask mandate')]

# Count the occurrences of each policy
policy_distribution = filtered_df['NPI measure'].value_counts()

# Count the number of policies implemented by each county (excluding those with only 'mask mandate')
county_policies_count = filtered_df.groupby('FIPS')['NPI measure'].nunique()

# Display the results
print(f"Policy distribution on and 14 days after {specific_date}:")
print(policy_distribution)

print(f"\nNumber of counties implementing different numbers of policies on and 14 days after {specific_date}:")
print(county_policies_count.value_counts())  # 113 counties in total

all_va_fips_pop = pd.read_csv('real_data/valid_counties_in_VA_population.csv')
all_va_fips = all_va_fips_pop.fips.values

va_adj = pd.read_csv('real_data/counties_adj_VA.csv')
selected_col_idx = [1, 3]
va_adj = va_adj.iloc[:, selected_col_idx]

all_umd_data = pd.read_csv('real_data/UMD-20210420-field_county.csv')
all_umd_data = all_umd_data[all_umd_data['fips'].isin(all_va_fips)]
all_umd_data['date'] = pd.to_datetime(all_umd_data['date'])
# selected date: 2020-04-15, 14 days after is 2020-04-29

start_date = pd.to_datetime(specific_date)
end_date = pd.to_datetime(date_plus_30)

all_umd_data = all_umd_data[(all_umd_data['date'] > start_date) & (all_umd_data['date'] <= end_date)]
all_umd_data_cases = all_umd_data[all_umd_data['field'] == 'New COVID cases']

cases_dict = dict()

for county in all_va_fips:
    sub_df = all_umd_data_cases[all_umd_data_cases['fips'] == county]
    all_new_cases_for_county = sub_df['value'].values.sum()
    cases_dict[county] = all_new_cases_for_county

T = county_policies_count.reset_index()
cases = pd.DataFrame.from_dict(cases_dict, orient='index').reset_index()
# Rename the columns
cases.columns = ['FIPS', 'Cases']
T_cases = cases.merge(T, on='FIPS', how='left')
T_cases = T_cases.fillna(0)
all_umd_data = pd.read_csv('real_data/UMD-20210420-field_county.csv')
features = all_umd_data[all_umd_data['date'] == '2020-04-15']
features = features[features['fips'].isin(all_va_fips)]
pivot_df = features.pivot(index='fips', columns='field', values='value')
pivot_df = pivot_df.reset_index()
pivot_df = pivot_df.merge(T_cases, left_on='fips', right_on='FIPS')
pivot_df.to_csv('2020-04-15-df-cases.csv')
