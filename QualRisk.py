import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Kathiir\coding bs\COVID_Data_Case_2021-05-24.csv")
countries_of_interest = [
    "United States", "Australia", "Canada", "France", "Germany",
    "Italy", "Japan", "Mexico", "Spain", "Switzerland"
]
df_filtered = df[df['location'].isin(countries_of_interest)]
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
# Group by 'location' and calculate the minimum and maximum dates
date_ranges = df_filtered.groupby('location')['date'].agg(['min', 'max'])
#print(date_ranges)
common_start = date_ranges['min'].max()  # Latest start date across the countries
common_end = date_ranges['max'].min()      # Earliest end date across the countries
df_common = df_filtered[(df_filtered['date'] >= common_start) & (df_filtered['date'] <= common_end)]
#print(f"Common date range in the dataset: {common_start.date()} to {common_end.date()}")

columns_to_drop_mortality = [
    'iso_code', 'continent', 'new_cases', 'new_deaths', 'new_cases_per_million',
    'new_deaths_per_million', 'reproduction_rate', 'icu_patients', 'icu_patients_per_million',
    'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions',
    'weekly_icu_admissions_per_million', 'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
    'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand',
    'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case', 
    'tests_units', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 
    'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred', 
    'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'new_vaccinations_smoothed_per_million',
    'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
    'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy',
    'human_development_index'
]

# Drop the columns from the original DataFrame and assign to a new variable
df_mortality = df_common.drop(columns=columns_to_drop_mortality)

us_df_mort = df_mortality[df_mortality['location'] == 'United States']
us_df_common = df_common[df_common['location'] == 'United States']

# Peak deaths per million and date
peak_row = us_df_mort.loc[us_df_mort['new_deaths_smoothed_per_million'].idxmax()]
peak_value = peak_row['new_deaths_smoothed_per_million']
peak_date = peak_row['date'].strftime("%B %Y")

# Average stringency index
avg_stringency_us = us_df_mort['stringency_index'].mean()

# Median age and % aged 65+
median_age_us = us_df_mort['median_age'].iloc[0]
aged_65_us = us_df_mort['aged_65_older'].iloc[0]
pop_dens_us = us_df_common['population_density'].iloc[0]
income_us = us_df_common['gdp_per_capita'].iloc[0]

# Print everything
print(f"📌 Peak deaths: {peak_value:.2f} per million in {peak_date}")
print(f"📌 Avg stringency index: {avg_stringency_us:.1f}%")
print(f"📌 Median age: {median_age_us} years")
print(f"📌 % aged 65+: {aged_65_us:.2f}%")
print(f"📌 Population density: {pop_dens_us} per square km")
print(f"📌 GDP per capita: ${income_us:.2f} ")

fig, ax1 = plt.subplots(figsize=(10, 5))

# First Y-axis: New deaths per million (left)
ax1.plot(us_df_mort['date'], us_df_mort['new_deaths_smoothed_per_million'], color='red', label='New Deaths per Million')
ax1.set_ylabel('New Deaths per Million', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Second Y-axis: Stringency index (right)
ax2 = ax1.twinx()
ax2.plot(us_df_mort['date'], us_df_mort['stringency_index'], color='blue', linestyle='--', label='Stringency Index')
ax2.set_ylabel('Stringency Index (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Formatting
plt.title("US COVID-19 Deaths vs. Stringency Index")
fig.tight_layout()
plt.show()

# Step 1: Filter only the 10 countries of interest
countries_of_interest = [
    "United States", "Australia", "Canada", "France", "Germany", 
    "Italy", "Japan", "Mexico", "Spain", "Switzerland"
]

df_filtered = df_mortality[df_mortality['location'].isin(countries_of_interest)]

# Step 2: Get the peak total_deaths_per_million for each country
peak_deaths = df_filtered.groupby('location')['total_deaths_per_million'].max()

# Step 3: Get % aged 65+ from any valid row (it's constant per country)
age_65 = df_filtered.groupby('location')['aged_65_older'].first()  # convert to %

# Step 4: Combine into a new DataFrame for plotting
summary_df = pd.DataFrame({
    'peak_deaths_per_million': peak_deaths,
    'aged_65_percent': age_65
}).sort_values(by='peak_deaths_per_million', ascending=False)

# Step 5: Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary axis: Peak deaths per million (bars)
bars = ax1.bar(summary_df.index, summary_df['peak_deaths_per_million'],
               color=['crimson' if country == 'United States' else 'steelblue' for country in summary_df.index])

ax1.set_ylabel("Peak Total Deaths per Million", color='black')
ax1.set_title("Peak COVID-19 Mortality vs. % Aged 65+ (by Country)")
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)

# Add numeric labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(i, height + 30, f"{int(height)}", ha='center', va='bottom', fontsize=9)

# Secondary axis: % aged 65+ (line)
ax2 = ax1.twinx()
ax2.scatter(summary_df.index, summary_df['aged_65_percent'], color='black', marker='o', zorder=3, label='% Aged 65+')
ax2.set_ylabel("% Aged 65+", color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0, 30)  # Adjust based on expected max % (e.g. Japan)

# Add labels for % aged 65+ above markers
for i, pct in enumerate(summary_df['aged_65_percent']):
    ax2.text(i, pct + 1, f"{pct:.1f}%", ha='center', va='bottom', fontsize=9, color='black')

plt.tight_layout()
plt.show()


columns = [
    'location', 'date', 'total_deaths_per_million', 'people_fully_vaccinated_per_hundred',
    'stringency_index', 'icu_patients_per_million', 'population_density', 'gdp_per_capita'
]

# Proceed with summary table generation again using placeholder df
countries = [
    "United States", "Australia", "Canada", "France", "Germany",
    "Italy", "Japan", "Mexico", "Spain", "Switzerland"
]

# Step 1: Get peak deaths per million for each country
peak_deaths = df_common.groupby('location')['total_deaths_per_million'].max()

# Step 2: Get the final available % fully vaccinated per country
latest_data = df_common.sort_values('date').groupby('location').tail(1)
vax_rate = latest_data.set_index('location')['people_fully_vaccinated_per_hundred']

# Step 3: Get average stringency index
avg_stringency = df_common.groupby('location')['stringency_index'].mean()

# Step 4: Get average ICU patients per million
avg_icu = df_common.groupby('location')['icu_patients_per_million'].mean()

# Step 5: Get population density
pop_density = df_common.groupby('location')['population_density'].first()

# Step 6: Get GDP per capita
gdp_per_capita = df_common.groupby('location')['gdp_per_capita'].first()

# Combine all metrics into a single summary table
summary_table = pd.DataFrame({
    'Peak Deaths per Million': peak_deaths,
    '% Fully Vaccinated': vax_rate,
    'Avg. Stringency Index': avg_stringency,
    'Avg. ICU Patients per Million': avg_icu,
    'Population Density': pop_density,
    'GDP per Capita ($)': gdp_per_capita
})

# Filter to countries of interest and round
summary_table = summary_table.loc[countries]
summary_table = summary_table.round({
    'Peak Deaths per Million': 0,
    '% Fully Vaccinated': 1,
    'Avg. Stringency Index': 0,
    'Avg. ICU Patients per Million': 1,
    'Population Density': 1,
    'GDP per Capita ($)': 0
})

summary_table.to_csv(r"C:\Kathiir\coding bs\covid_summary_table.csv", index=True)
print(summary_table)


# Total number of cells in df_common
total_cells = df_common.shape[0] * df_common.shape[1]

# Total number of missing values
missing_total = df_common.isna().sum().sum()
missing_percent_total = (missing_total / total_cells) * 100

print(f"🔍 Overall missing data: {missing_total} missing values out of {total_cells} "
      f"({missing_percent_total:.2f}% missing)")

# Column-by-column % missing
missing_by_col = df_common.isna().mean().sort_values(ascending=False) * 100
missing_by_col = missing_by_col[missing_by_col > 0]  # filter only columns with missing data

# Display top columns with missing data
print("\n📊 Missing data by column (percent):")
print(missing_by_col.round(1).to_string())

# Changed for each target country (GER, JAP, MEX)
df_mortality_jap = df_common.drop(columns=columns_to_drop_mortality)

jap_df_mort = df_mortality[df_mortality['location'] == 'Japan']
jap_df_common = df_common[df_common['location'] == 'Japan']

# Peak deaths per million and date
peak_row = jap_df_mort.loc[jap_df_mort['new_deaths_smoothed_per_million'].idxmax()]
peak_value = peak_row['new_deaths_smoothed_per_million']
peak_date = peak_row['date'].strftime("%B %Y")

# Average stringency index
avg_stringency_us = jap_df_mort['stringency_index'].mean()

# Median age and % aged 65+
median_age_us = jap_df_mort['median_age'].iloc[0]
aged_65_us = jap_df_mort['aged_65_older'].iloc[0]
pop_dens_us = jap_df_common['population_density'].iloc[0]
income_us = jap_df_common['gdp_per_capita'].iloc[0]

# Print everything
print(f"📌 Peak deaths: {peak_value:.2f} per million in {peak_date}")
print(f"📌 Avg stringency index: {avg_stringency_us:.1f}%")
print(f"📌 Median age: {median_age_us} years")
print(f"📌 % aged 65+: {aged_65_us:.2f}%")
print(f"📌 Population density: {pop_dens_us} per square km")
print(f"📌 GDP per capita: ${income_us:.2f} ")

fig, ax1 = plt.subplots(figsize=(10, 5))

# First Y-axis: New deaths per million (left)
ax1.plot(jap_df_mort['date'], jap_df_mort['new_deaths_smoothed_per_million'], color='red', label='New Deaths per Million')
ax1.set_ylabel('New Deaths per Million', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Second Y-axis: Stringency index (right)
ax2 = ax1.twinx()
ax2.plot(jap_df_mort['date'], jap_df_mort['stringency_index'], color='blue', linestyle='--', label='Stringency Index')
ax2.set_ylabel('Stringency Index (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Formatting
plt.title("Japan COVID-19 Deaths vs. Stringency Index")
fig.tight_layout()
plt.show()