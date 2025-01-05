import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import geopandas as gpd
!pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns
import requests
import warnings
warnings.filterwarnings('ignore')

#reading our csv dataset using pandas
landslide_df=pd.read_csv('Global_Landslide_Catalog_Export.csv')
#display the dataframe overview
landslide_df

#As we have all empty values in event_time column we can drop it
landslide_df.drop(columns=['event_time'], inplace=True)

#event id column has unique values in the dataset,
#we can use that attribute as primary key for further operations. lets move that column to the first place and rearrange all the records in sorted order by event_id
columns = landslide_df.columns.tolist()
columns.insert(0, columns.pop(2))
landslide_df=landslide_df[columns]
landslide_df = landslide_df.sort_values(by='event_id')

# filter landslide data for US for further uses below.
us_data = landslide_df[landslide_df['country_name'] == 'United States']
# Filter landslide data for India for further uses below.
india_data = landslide_df[landslide_df['country_name'] == 'India']

#us_data
#displaying how the indian landslides data look like after perfroming above mentioned changes
india_data

#printing the datatypes of all the attributes in the dataset
landslide_df.dtypes

#performing some basic statisctics on numerical data in the dataset
landslide_df.describe()

#this non zero values tell the count of empty cells in the given data.
print(landslide_df.isnull().sum())

# Get a concise summary of the dataframe
print(landslide_df.info())

#these below lines helps us to show the rows having the null values in their records
rows_with_empty_values = landslide_df[landslide_df.isnull().any(axis=1)]
rows_with_empty_values

# Descriptive statistics, This won't provide much information about dataset we are working
print(landslide_df.describe())

# Histograms for numeric data
landslide_df.hist(figsize=(15, 10))
plt.show()

# Convert 'event_date' to datetime format
landslide_df['event_date'] = pd.to_datetime(landslide_df['event_date'], format='%m/%d/%Y %I:%M:%S %p')

# Extract year, month, and day components
landslide_df['year'] = landslide_df['event_date'].dt.year
landslide_df['month'] = landslide_df['event_date'].dt.month
landslide_df['day'] = landslide_df['event_date'].dt.day

#let's plot the a bar plot showing the landslides events happened by year in the world
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df)
plt.title('Landslide Events by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#let's plot the a bar plot showing the landslides events happened by year in United States
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df[landslide_df['country_name'] == 'United States'])
plt.title('Landslide Events by Year in United States')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#let's plot the a bar plot showing the landslides events happened by year in India

plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=landslide_df[landslide_df['country_name'] == 'India'])
plt.title('Landslide Events by Year in India')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#as we have various attributes with numerical values lets box plot them which shows the distribution of values as per the dataset
numerical_data = landslide_df.select_dtypes(include=['float64', 'int64'])

# Define the number of plots along the x and y axis
n_cols = 3  # Number of columns in the subplot grid
n_rows = (numerical_data.shape[1] + n_cols - 1) // n_cols  # Calculate number of rows needed

# Create a figure and axes with a dynamic size based on the number of subplots
plt.figure(figsize=(n_cols * 5, n_rows * 4))  # Adjust size dynamically

# Create subplots for each numerical column
for i, column in enumerate(numerical_data.columns):
    plt.subplot(n_rows, n_cols, i + 1)  # Position each subplot in the correct location
    landslide_df.boxplot(column=[column])
    plt.title(column)

plt.suptitle('Box Plots of Numerical Variables')
plt.tight_layout(pad=3.0)  # Add padding to ensure subplots don't overlap
plt.show()

storm_landslides = landslide_df[landslide_df['storm_name'].notna()]
# Summarizing the data
storm_summary = storm_landslides[['storm_name', 'landslide_size', 'fatality_count', 'injury_count']].groupby(['storm_name', 'landslide_size']).agg(
    total_events=pd.NamedAgg(column='storm_name', aggfunc='count'),
    total_fatalities=pd.NamedAgg(column='fatality_count', aggfunc='sum'),
    total_injuries=pd.NamedAgg(column='injury_count', aggfunc='sum')
).reset_index()
storm_summary

# Visualizing the relationship between storms and landslide size
plt.figure(figsize=(14, 8))
sns.countplot(data=storm_landslides, y='storm_name', hue='landslide_size', order=storm_landslides['storm_name'].value_counts().iloc[:10].index)
#sns.countplot(data=storm_landslides, y='storm_name', hue='landslide_size', order=storm_landslides['storm_name'].value_counts().iloc[:20].index)
plt.title('Frequency of Landslide Events by Storm and Size of Landslide')
plt.xlabel('Number of Events')
plt.ylabel('Storm Name')
plt.legend(title='Landslide Size')
plt.show()

import matplotlib.pyplot as plt

# Assuming your dataset is stored in a DataFrame called 'df' and the attribute you're interested in is called 'attribute_name'

# Calculate the frequency of each value
value_counts = landslide_df['landslide_size'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(6, 6))  # Increase the figure size
plt.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', startangle=180, labeldistance=1.1)  # Adjust labeldistance
plt.title('Landslide sizes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Convert the 'event_date' column to datetime format
landslide_df['event_date'] = pd.to_datetime(landslide_df['event_date'], format='%m/%d/%Y %I:%M:%S %p')

# Drop any rows with NaT values in the 'event_date' column
landslide_df.dropna(subset=['event_date'], inplace=True)

# Set the 'event_date' as the index of the DataFrame
landslide_df.set_index('event_date', inplace=True)

# Resampling the data yearly and counting the number of events per year
yearly_data = landslide_df.resample('Y').size()

# Plotting the time series of landslide events
plt.figure(figsize=(10, 6))
yearly_data.plot(title='Landslide Events per Year', marker='*')
plt.xlabel('Year')
plt.ylabel('Number of Landslide Events')
plt.grid(True)
plt.show()

# Optional: Decompose the time series to observe trend, seasonal, and residual components
decomposition = seasonal_decompose(yearly_data, model='additive', period=2)  # Adjust period according to dataset
fig = decomposition.plot()
plt.show()

# Select numerical columns for correlation
numerical_columns = ['year', 'month', 'day', 'fatality_count', 'injury_count']  # Add other numerical columns as needed
numerical_df = landslide_df[numerical_columns]

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()


# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.5f', cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

def landslide_visualization(landslide_df):
    """
    Visualizes the global occurrence of landslide events using a map plot.

    Parameters:
    - landslide_df (DataFrame): DataFrame containing landslide data with columns 'latitude' and 'longitude'.

    Returns:
    - None
    """

    try:
        # Create GeoDataFrame with the latitude and longitude coordinates
        global_map = gpd.GeoDataFrame(landslide_df, geometry=gpd.points_from_xy(landslide_df['longitude'], landslide_df['latitude']))

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_title('Global Occurrence of Landslide Events', fontsize=20)

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgrey')

        # Plot the landslide events
        global_map.plot(ax=ax, marker='o', color='green', markersize=2, alpha=0.8, label='Landslide Events')

        # Add legend
        ax.legend(loc='upper left', fontsize=12)

        # Add compass rose
        ax.annotate('N', xy=(0.5, 1), xytext=(0.5, 1.05), xycoords='axes fraction', fontsize=12, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-|>', color='black'))

        # Add gridlines
        ax.gridlines(draw_labels=True, linestyle='--')

        # Adjust the extent of the map
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

        # Show the plot
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("An error occurred:", e)

landslide_visualization(landslide_df)
#landslide_visualization(us_data)
#landslide_visualization(india_data)

import plotly.graph_objects as go

# Create a 3D Earth map
fig = go.Figure()

# Add a scattergeo trace
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

# Update layout for 3D view
fig.update_geos(
    projection_type='orthographic',
    showland=True,
    showocean=True,
    oceancolor='lightblue',
    landcolor='lightgrey',
    showcountries=True
)

# Set title
fig.update_layout(title='3D Earth Map of Landslide Events')

# Show the plot
fig.show()

# Create GeoDataFrame with the latitude and longitude coordinates
global_map = gpd.GeoDataFrame(landslide_df, geometry=gpd.points_from_xy(landslide_df['longitude'], landslide_df['latitude']))

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title('Global Distribution of Landslide Events', fontsize=20)

# Add background map
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)

# Plot latitude and longitude using hexbin
hb = ax.hexbin(global_map['longitude'], global_map['latitude'], gridsize=50, cmap='YlOrRd', alpha=0.8)

# Add colorbar
cb = plt.colorbar(hb, ax=ax)
cb.set_label('Number of Landslide Events')

# Add gridlines with labels
ax.gridlines(draw_labels=True, linestyle='--')

# Adjust the extent of the map
ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

# Show the plot
plt.tight_layout()
plt.show()

import plotly.graph_objects as go

# Create a scatter plot
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

# Set axis titles
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'
                    ),
                  title='3D Scatter Plot of Landslide Events')

# Show the plot
fig.show()

us_data['event_date'] = pd.to_datetime(us_data['event_date'], format='%m/%d/%Y %I:%M:%S %p')
#us_data['event_date'] = pd.to_datetime(us_data['event_date'], errors='coerce')
us_data['year'] = us_data['event_date'].dt.year
us_data['month'] = us_data['event_date'].dt.month
us_data['day'] = us_data['event_date'].dt.day

india_data['event_date'] = pd.to_datetime(india_data['event_date'], format='%m/%d/%Y %I:%M:%S %p')
india_data['year'] = india_data['event_date'].dt.year
india_data['month'] = india_data['event_date'].dt.month
india_data['day'] = india_data['event_date'].dt.day

# Extracting the month from the 'event_date' column to find the season
# This seasoning is done as per USA seasonal patterns
month_to_season = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
}
us_data['season'] = us_data['month'].map(month_to_season)

# Count the number of landslides in each season
landslide_season_counts = us_data['season'].value_counts()

# Plot the distribution of landslides by season
landslide_season_counts.plot(kind='bar')
plt.title('Number of Landslides by Season')
plt.xlabel('Season')
plt.ylabel('Number of Landslides')
plt.show()

 Perform a Chi-squared test for US data to see if the distribution across seasons is uniform
observed_frequencies = landslide_season_counts.values
expected_frequencies = [len(us_data) / 4] * 4  # Expect uniform distribution over 4 seasons
chi2, p_value = chi2_contingency([observed_frequencies, expected_frequencies])[0:2]

# Output the results
print(f"Observed Frequencies: {observed_frequencies}")
print(f"Expected Frequencies (Uniform Distribution): {expected_frequencies}")
print(f"Chi-squared Test Statistic: {chi2}")
print(f"P-value: {p_value}")

# Extracting the month from the 'event_date' column to find the season
# This seasoning is done as per Indian time zones
month_to_season = {
    1: 'Winter', 2: 'Winter',
    3: 'Summer', 4: 'Summer', 5: 'Summer',
    6: 'Summer', 7: 'Monsoon', 8: 'Monsoon',
    9: 'Monsoon', 10: 'Monsoon', 11: 'Winter', 12: 'Winter'
}
india_data['season'] = india_data['month'].map(month_to_season)

# Count the number of landslides in each season
landslide_season_counts = india_data['season'].value_counts()



# Plot the distribution of landslides by season
landslide_season_counts.plot(kind='bar')
plt.title('Number of Landslides by Season')
plt.xlabel('Season')
plt.ylabel('Number of Landslides')
plt.show()

# Perform a Chi-squared test to see if the distribution across seasons is uniform
observed_frequencies = landslide_season_counts.values
expected_frequencies = [len(india_data) / 3] * 3  # Expect uniform distribution over 4 seasons
chi2, p_value = chi2_contingency([observed_frequencies, expected_frequencies])[0:2]

# Output the results
print(f"Observed Frequencies: {observed_frequencies}")
print(f"Expected Frequencies (Uniform Distribution): {expected_frequencies}")
print(f"Chi-squared Test Statistic: {chi2}")
print(f"P-value: {p_value}")
