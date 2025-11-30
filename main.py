import pandas as pd
import numpy as np
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans
import os
import sys

# Attempt to locate data_output.csv in common locations; if not found, create a small synthetic DataFrame for development/testing.
data_paths = [
    os.path.join(os.getcwd(), "data_output.csv"),
    os.path.join(os.getcwd(), "data", "data_output.csv"),
]
# If running from a script, __file__ can help locate the project directory
if '__file__' in globals():
    data_paths.insert(0, os.path.join(os.path.dirname(__file__), "data_output.csv"))

df = None
for p in data_paths:
    try:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            print(f"Loaded data from {p}")
            break
    except Exception as e:
        print(f"Found {p} but failed to read: {e}")

if df is None:
    print("Warning: data_output.csv not found. Creating a small synthetic dataset for development/testing.")
    n = 50
    now = pd.Timestamp.now()
    df = pd.DataFrame({
        'tpep_pickup_datetime': [now + pd.Timedelta(minutes=15*i) for i in range(n)],
        'tpep_dropoff_datetime': [now + pd.Timedelta(minutes=15*i+10) for i in range(n)],
        'trip_distance': np.random.rand(n)*10,
        'passenger_count': np.random.randint(1,5,size=n),
        'congestion_surcharge': np.random.rand(n)*2,
        'PULocationID': np.random.randint(1, 50, size=n),
        'DOLocationID': np.random.randint(1, 50, size=n),
    })

# Data Preprocessing
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
df['Region'] = np.random.randint(1, 5, size=len(df))

X = df[['hour', 'day_of_week', 'trip_distance', 'passenger_count']]
y = df['congestion_surcharge']

nan_indices_y = np.isnan(y)
if np.any(nan_indices_y):
    X = X[~nan_indices_y]
    y = y[~nan_indices_y]

nan_indices_X = np.isnan(X)
if np.any(nan_indices_X):
    X = X[~nan_indices_X]
    y = y[~nan_indices_X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()

nan_indices_X_train = np.isnan(X_train)
nan_indices_y_train = np.isnan(y_train)
if np.any(nan_indices_X_train) or np.any(nan_indices_y_train):
    X_train = X_train[~(nan_indices_X_train | nan_indices_y_train)]
    y_train = y_train[~(nan_indices_X_train | nan_indices_y_train)]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (Demand Prediction):", mae)


def pickup_in_day(frame):
    hour_pickups = []
    temp = []
    for i in range(7):
        for j in range(24):
            temp.append(frame[(frame.day_of_week == i) & (frame.hour == j)].shape[0])
        hour_pickups.append(temp)
        temp = []

    colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:brown', 'xkcd:coral', 'xkcd:magenta', 'xkcd:green', 'xkcd:fuchsia']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    plt.figure(figsize=(8, 4))
    hours_lis = [s for s in range(0, 24)]
    for k in range(0, 7):
        plt.plot(hours_lis, hour_pickups[k], colors[k], label=days[k])
        plt.plot(hours_lis, hour_pickups[k], 'ro', markersize=2)

    plt.xticks([s for s in range(0, 24)])
    plt.xlabel('Hours of a day')
    plt.ylabel('Number of pickups')
    plt.title('Pickups for every hour')
    plt.legend()
    plt.grid(True)
    plt.show()

    hour_pickup_month = []
    for j in range(0, 24):
        hour_pickup_month.append(frame[frame.hour == j].shape[0])

    plt.figure(figsize=(8, 4))
    hours_lis = [s for s in range(0, 24)]
    plt.plot(hours_lis, hour_pickup_month, 'xkcd:magenta', label='average pickups per hour')
    plt.plot(hours_lis, hour_pickup_month, 'ro', markersize=2)

    plt.xticks([s for s in range(0, 24)])
    plt.xlabel('Hours of a day')
    plt.ylabel('Number of pickups')
    plt.title('Pickups for every hour for the whole month')
    plt.legend()
    plt.grid(True)
    plt.show()


pickup_in_day(df)


def Convert_Clusters(frame, cluster):
    region = []
    colors = []
    Queens = [4, 16, 22, 23, 26, 35, 36]
    Brooklyn = [11, 19, 29, 37]
    for i in frame[cluster].values:
        if i == 2:
            region.append("JFK")
            colors.append('#7CFC00')  # Green - JFK
        elif i == 13:
            region.append("Bronx")
            colors.append('#DC143C')  # Red - Bronx
        elif i in Queens:
            region.append("Queens")
            colors.append('#00FFFF')  # Blue - Queens
        elif i in Brooklyn:
            region.append("Brooklyn")
            colors.append('#FFD700')  # Brooklyn - yellow orange
        else:
            region.append("Manhattan")
            colors.append('#FFFFFF')  # White - Manhattan
    frame['Regions ' + cluster] = region
    return frame, colors


print(df.columns)

X_pickup = df[['PULocationID', 'DOLocationID']]
# Ensure no missing rows are dropped out-of-band (keep alignment with df)
X_pickup = X_pickup.fillna(0)

kmeans = KMeans(n_clusters=5, random_state=42)
# Fit on the full (filled) X_pickup and assign labels for every row
df['pickup_cluster'] = kmeans.fit_predict(X_pickup)

frame_with_clusters, colors_pickup = Convert_Clusters(df, 'pickup_cluster')


# Plot Scatter Locations
def plot_scatter_locations(frame, colors, choose):
    plt.style.use('default')  # Better Styling
    rcParams['figure.figsize'] = (12, 12)  # Size of figure
    rcParams['figure.dpi'] = 100

    # Create a GeoDataFrame from the DataFrame
    geometry = [Point(xy) for xy in zip(frame['PULocationID'], frame['DOLocationID'])]
    gdf = gpd.GeoDataFrame(frame, geometry=geometry)

    # Check CRS
    if gdf.crs is None:
        print("Warning: GeoDataFrame CRS is not defined.")
        # You might want to set a CRS if it's not already set. For example:
        # gdf.crs = 'EPSG:4326'  # Replace with the appropriate CRS

    # Plot NYC basemap
    ax = gdf.plot(markersize=5, color=colors, alpha=0.7)

    # Set aspect ratio manually
    ax.set_aspect('equal')

    # Check if CRS is set before using to_string()
    if gdf.crs is not None:
        ctx.plotting.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite)

    plt.title(f'{choose.capitalize()} Locations on NYC Map')
    plt.show()


plot_scatter_locations(frame_with_clusters, colors_pickup, 'pickup_cluster')

frame_with_durations_outliers_removed, colors_pickup = Convert_Clusters(df, 'pickup_cluster')
plot_scatter_locations(frame_with_durations_outliers_removed, colors_pickup, 'pickup_cluster')


# Define coordinates for Manhattan clusters
cluster1 = [(40.770, -73.980), (40.760, -73.980), (40.760, -73.970), (40.770, -73.970)]  # Midtown
cluster2 = [(40.730, -74.010), (40.720, -74.010), (40.720, -73.990), (40.730, -73.990)]  # Financial District
cluster3 = [(40.780, -73.950), (40.770, -73.950), (40.770, -73.930), (40.780, -73.930)]  # Upper East Side

# Extract latitude and longitude for plotting
cluster1_lat, cluster1_lon = zip(*cluster1)
cluster2_lat, cluster2_lon = zip(*cluster2)
cluster3_lat, cluster3_lon = zip(*cluster3)

# Plot Manhattan clusters
plt.scatter(cluster1_lon, cluster1_lat, c='red', label='Midtown')
plt.scatter(cluster2_lon, cluster2_lat, c='blue', label='Financial District')
plt.scatter(cluster3_lon, cluster3_lat, c='green', label='Upper East Side')

# Customize the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Manual Cluster Plot for Manhattan')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

locations = df[['PULocationID', 'DOLocationID']].values


def distance_function(route):
    total_distance = 0
    for i in range(len(route) - 1):
        start_location = locations[route[i]]
        end_location = locations[route[i + 1]]
        distance = np.linalg.norm(end_location - start_location)
        total_distance += distance
    return total_distance,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, len(locations))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", distance_function)

population = toolbox.population(n=2)
hof = tools.HallOfFame(1)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=2, stats=None, halloffame=hof, verbose=True)

best_individual = hof[0]
optimized_route = best_individual
print("Optimized Route:", optimized_route)
