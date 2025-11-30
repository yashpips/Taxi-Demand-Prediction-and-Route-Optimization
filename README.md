# NYC Taxi Demand Prediction and Route Optimization

This repository contains code for predicting taxi demand in New York City (NYC) and optimizing taxi routes using Python. The project utilizes machine learning techniques such as Random Forest Regression for demand prediction and Genetic Algorithm for route optimization. The dataset used for analysis is a CSV file containing NYC taxi trip records.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Visualization](#visualization)
7. [Route Optimization](#route-optimization)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

The goal of this project is to predict taxi demand in different regions of NYC and optimize taxi routes for efficient transportation. The project uses machine learning models for demand prediction and genetic algorithms for route optimization.

## Installation

To run the code in this repository, you need to have Python installed on your system. Additionally, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- contextily
- geopandas
- shapely
- deap

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn contextily geopandas shapely deap
```

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yashpips/Taxi-Demand-Prediction-and-Route-Optimization.git
```

2. Navigate to the project directory:

```bash
cd Taxi-Demand-Prediction-and-Route-Optimization
```

3. Run the main script to execute the entire pipeline:

```bash
python main.py
```

## Data Preprocessing

The dataset (`data_output.csv`) contains taxi trip records with pickup and drop-off locations, timestamps, trip distance, and passenger count. Data preprocessing steps include converting timestamps, extracting features like hour and day of week, and handling missing values.

## Model Training

The project uses a Random Forest Regressor to predict taxi demand based on features such as hour, day of week, trip distance, and passenger count. The model is trained on a subset of the dataset and evaluated using Mean Absolute Error (MAE).

## Visualization

Visualizations are created to analyze taxi demand trends, pickup locations, and cluster regions in NYC. The `pickup_in_day` function plots the number of pickups for each hour of the day and day of the week. Scatter plots on NYC maps visualize pickup locations and cluster regions.

## Route Optimization

The project includes a genetic algorithm for optimizing taxi routes. The `distance_function` calculates the total distance of a given route based on pickup and drop-off locations. The genetic algorithm optimizes routes to minimize total distance traveled.

## Contributing

Contributions to this project are welcome. You can contribute by adding new features, improving existing code, or fixing issues. Please fork the repository, make your changes, and submit a pull request.

