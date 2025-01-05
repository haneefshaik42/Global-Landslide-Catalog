import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import geopandas as gpd
get_ipython().system('pip install cartopy')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns
import requests
import warnings
warnings.filterwarnings('ignore')
landslide_df=pd.read_csv('Global_Landslide_Catalog_Export.csv')
landslide_df
landslide_df.drop(columns=['event_time'], inplace=True)
columns = landslide_df.columns.tolist()
columns.insert(0, columns.pop(2))
landslide_df=landslide_df[columns]
landslide_df = landslide_df.sort_values(by='event_id')
us_data = landslide_df[landslide_df['country_name'] == 'United States']
india_data = landslide_df[landslide_df['country_name'] == 'India']
india_data
landslide_df.dtypes
landslide_df.describe()
print(landslide_df.isnull().sum())
print(landslide_df.info())
rows_with_empty_values = landslide_df[landslide_df.isnull().any(axis=1)]
rows_with_empty_values
print(landslide_df.describe())
landslide_df.hist(figsize=(15, 10))
plt.show()
landslide_df['event_date'] = pd.to_datetime(landslide_df['event_date'], format='%m/%d/%Y %I:%M:%S %p')
landslide_df['year'] = landslide_df['event_date'].dt.year
landslide_df['month'] = landslide_df['event_date'].dt.month
landslide_df['day'] = landslide_df['event_date'].dt.day
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df)
plt.title('Landslide Events by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df[landslide_df['country_name'] == 'United States'])
plt.title('Landslide Events by Year in United States')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df[landslide_df['country_name'] == 'India'])
plt.title('Landslide Events by Year in India')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
numerical_data = landslide_df.select_dtypes(include=['float64', 'int64'])
n_cols = 3  # Number of columns in the subplot grid
n_rows = (numerical_data.shape[1] + n_cols - 1) // n_cols  # Calculate number of rows needed
plt.figure(figsize=(n_cols * 5, n_rows * 4))  # Adjust size dynamically
for i, column in enumerate(numerical_data.columns):
    plt.subplot(n_rows, n_cols, i + 1)  # Position each subplot in the correct location
    landslide_df.boxplot(column=[column])
    plt.title(column)
plt.suptitle('Box Plots of Numerical Variables')
plt.tight_layout(pad=3.0)  # Add padding to ensure subplots don't overlap
plt.show()
storm_landslides = landslide_df[landslide_df['storm_name'].notna()]
storm_summary = storm_landslides[['storm_name', 'landslide_size', 'fatality_count', 'injury_count']].groupby(['storm_name', 'landslide_size']).agg(
    total_events=pd.NamedAgg(column='storm_name', aggfunc='count'),
    total_fatalities=pd.NamedAgg(column='fatality_count', aggfunc='sum'),
    total_injuries=pd.NamedAgg(column='injury_count', aggfunc='sum')
).reset_index()
storm_summary
plt.figure(figsize=(14, 8))
sns.countplot(data=storm_landslides, y='storm_name', hue='landslide_size', order=storm_landslides['storm_name'].value_counts().iloc[:10].index)
plt.title('Frequency of Landslide Events by Storm and Size of Landslide')
plt.xlabel('Number of Events')
plt.ylabel('Storm Name')
plt.legend(title='Landslide Size')
plt.show()
import matplotlib.pyplot as plt
value_counts = landslide_df['landslide_size'].value_counts()
plt.figure(figsize=(6, 6))  # Increase the figure size
plt.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', startangle=180, labeldistance=1.1)  # Adjust labeldistance
plt.title('Landslide sizes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
landslide_df['event_date'] = pd.to_datetime(landslide_df['event_date'], format='%m/%d/%Y %I:%M:%S %p')
landslide_df.dropna(subset=['event_date'], inplace=True)
landslide_df.set_index('event_date', inplace=True)
yearly_data = landslide_df.resample('Y').size()
plt.figure(figsize=(10, 6))
yearly_data.plot(title='Landslide Events per Year', marker='*')
plt.xlabel('Year')
plt.ylabel('Number of Landslide Events')
plt.grid(True)
plt.show()
decomposition = seasonal_decompose(yearly_data, model='additive', period=2)  # Adjust period according to dataset
fig = decomposition.plot()
plt.show()
numerical_columns = ['year', 'month', 'day', 'fatality_count', 'injury_count']  # Add other numerical columns as needed
numerical_df = landslide_df[numerical_columns]
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.5f', cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
def landslide_visualization(landslide_df):
    try:
        global_map = gpd.GeoDataFrame(landslide_df, geometry=gpd.points_from_xy(landslide_df['longitude'], landslide_df['latitude']))
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_title('Global Occurrence of Landslide Events', fontsize=20)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgrey')
        global_map.plot(ax=ax, marker='o', color='green', markersize=2, alpha=0.8, label='Landslide Events')
        ax.legend(loc='upper left', fontsize=12)
        ax.annotate('N', xy=(0.5, 1), xytext=(0.5, 1.05), xycoords='axes fraction', fontsize=12, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-|>', color='black'))
        ax.gridlines(draw_labels=True, linestyle='--')
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("An error occurred:", e)
landslide_visualization(landslide_df)
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scattergeo(
    lon=landslide_df['longitude'],
    lat=landslide_df['latitude'],
    mode='markers',
    marker=dict(
        size=2,
        color='darkgreen',
        opacity=0.8
    )
))
fig.update_geos(
    projection_type='orthographic',
    showland=True,
    showocean=True,
    oceancolor='lightblue',
    landcolor='lightgrey',
    showcountries=True
)
fig.update_layout(title='3D Earth Map of Landslide Events')
fig.show()
global_map = gpd.GeoDataFrame(landslide_df, geometry=gpd.points_from_xy(landslide_df['longitude'], landslide_df['latitude']))
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title('Global Distribution of Landslide Events', fontsize=20)
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
hb = ax.hexbin(global_map['longitude'], global_map['latitude'], gridsize=50, cmap='YlOrRd', alpha=0.8)
cb = plt.colorbar(hb, ax=ax)
cb.set_label('Number of Landslide Events')
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
plt.tight_layout()
plt.show()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter3d(
    x=landslide_df['longitude'],
    y=landslide_df['latitude'],
    z=landslide_df.index,  # Assuming the index represents time
    mode='markers',
    marker=dict(
        size=2,
        color='green',
        opacity=0.8
    )
))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'
                    ),
                  title='3D Scatter Plot of Landslide Events')
fig.show()
us_data['event_date'] = pd.to_datetime(us_data['event_date'], format='%m/%d/%Y %I:%M:%S %p')
us_data['year'] = us_data['event_date'].dt.year
us_data['month'] = us_data['event_date'].dt.month
us_data['day'] = us_data['event_date'].dt.day
india_data['event_date'] = pd.to_datetime(india_data['event_date'], format='%m/%d/%Y %I:%M:%S %p')
india_data['year'] = india_data['event_date'].dt.year
india_data['month'] = india_data['event_date'].dt.month
india_data['day'] = india_data['event_date'].dt.day
month_to_season = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
}
us_data['season'] = us_data['month'].map(month_to_season)
landslide_season_counts = us_data['season'].value_counts()
landslide_season_counts.plot(kind='bar')
plt.title('Number of Landslides by Season')
plt.xlabel('Season')
plt.ylabel('Number of Landslides')
plt.show()
observed_frequencies = landslide_season_counts.values
expected_frequencies = [len(us_data) / 4] * 4  # Expect uniform distribution over 4 seasons
chi2, p_value = chi2_contingency([observed_frequencies, expected_frequencies])[0:2]
print(f"Observed Frequencies: {observed_frequencies}")
print(f"Expected Frequencies (Uniform Distribution): {expected_frequencies}")
print(f"Chi-squared Test Statistic: {chi2}")
print(f"P-value: {p_value}")
month_to_season = {
    1: 'Winter', 2: 'Winter',
    3: 'Summer', 4: 'Summer', 5: 'Summer',
    6: 'Summer', 7: 'Monsoon', 8: 'Monsoon',
    9: 'Monsoon', 10: 'Monsoon', 11: 'Winter', 12: 'Winter'
}
india_data['season'] = india_data['month'].map(month_to_season)
landslide_season_counts = india_data['season'].value_counts()
landslide_season_counts.plot(kind='bar')
plt.title('Number of Landslides by Season')
plt.xlabel('Season')
plt.ylabel('Number of Landslides')
plt.show()
observed_frequencies = landslide_season_counts.values
expected_frequencies = [len(india_data) / 3] * 3  # Expect uniform distribution over 4 seasons
chi2, p_value = chi2_contingency([observed_frequencies, expected_frequencies])[0:2]
print(f"Observed Frequencies: {observed_frequencies}")
print(f"Expected Frequencies (Uniform Distribution): {expected_frequencies}")
print(f"Chi-squared Test Statistic: {chi2}")
print(f"P-value: {p_value}")
