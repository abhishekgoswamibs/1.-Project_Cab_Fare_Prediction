# Importing the liabraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
#% matplotlib inline 

#setting working directory
os.chdir("D:/Data Science Edwisor/6. Projects/1. Project One Cab Rental Company")

# Importing the dataset
dataset = pd.read_csv('train_cab.csv')

# Exploratory Data Analysis
# Converting to proper data type
dataset['fare_amount'] = pd.to_numeric(dataset['fare_amount'], errors = 'coerce')
dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], errors = 'coerce')
dataset['passenger_count'] = dataset['passenger_count'].astype('object')
# Extracting useful information from 'pickup_datetime'
dataset['pickup_date'] = dataset['pickup_datetime'].dt.day
dataset['pickup_day'] = dataset['pickup_datetime'].dt.weekday
dataset['pickup_month'] = dataset['pickup_datetime'].dt.month
dataset['pickup_hour'] = dataset['pickup_datetime'].dt.hour
dataset['pickup_minute'] = dataset['pickup_datetime'].dt.minute
dataset['pickup_day'] = dataset['pickup_day'].astype('object')
dataset['pickup_hour'] = dataset['pickup_hour'].replace(0, 24)

# Removing pickup_datetime variable from dataset
dataset = dataset.iloc[:, [0,2,3,4,5,6,7,8,9,10,11]]

# Inspecting the Latitudes and longitudes
# Latitude(-90 - 90)
dataset[dataset['pickup_latitude'] < -90]
dataset[dataset['pickup_latitude'] > 90]
dataset = dataset.drop(dataset[dataset['pickup_latitude'] > 90].index, axis = 0)

dataset[dataset['dropoff_latitude'] < -90]
dataset[dataset['dropoff_latitude'] > 90]

# Longitude(-180 - 180)
dataset[dataset['pickup_longitude'] < -180]
dataset[dataset['pickup_longitude'] > 180]

dataset[dataset['dropoff_longitude'] < -180]
dataset[dataset['dropoff_longitude'] > 180]


# Data Preprocessing

# 1. Missing Value Analysis.
dataset.isnull().sum()

"""# Finding the best method for missing value imputation.
dataset['pickup_longitude'].loc[7]
# Actual Value is -73.9513
dataset['pickup_longitude'].loc[7] = np.nan

# Mean
dataset['pickup_longitude'] = dataset['pickup_longitude'].fillna(dataset['pickup_longitude'].mean())
# Mean value = -73.912

# Median
dataset['pickup_longitude'] = dataset['pickup_longitude'].fillna(dataset['pickup_longitude'].median())
# Median value = -73.982

# KNN
from fancyimpute import KNN
imputer = KNN(k = 5)
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns = dataset.columns)
# KNN Value = -73.993
# Imputing with median
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(dataset.iloc[:, [0,1,2,3,4,6,8,9,10]])
dataset.iloc[:, [0,1,2,3,4,6,8,9,10]] = imputer.transform(dataset.iloc[:, [0,1,2,3,4,6,8,9,10]])
# Imputing with mode
imputer_mode = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_mode = imputer_mode.fit(dataset.iloc[:, [5,7]])
dataset.iloc[:, [5,7]] = imputer_mode.transform(dataset.iloc[:, [5,7]])
dataset.isnull().sum()
dataset['passenger_count'] = dataset['passenger_count'].astype('object')
dataset['pickup_day'] = dataset['pickup_day'].astype('object')"""
# Dropping the values with NAN. 
dataset = dataset.drop(dataset[dataset['pickup_date'].isnull()].index, axis = 0)
dataset = dataset.drop(dataset[dataset['fare_amount'].isnull()].index, axis = 0)
dataset = dataset.drop(dataset[dataset['passenger_count'].isnull()].index, axis = 0)
dataset.isnull().sum()
# 2. Outiler Analysis.

# 1. Fare_amount
dataset['fare_amount'].describe()
dataset['fare_amount'].sort_values(ascending = False)
# Some very high values present in fare_amount
# Removing values > 150
dataset = dataset.drop(dataset[dataset['fare_amount'] > 150].index, axis = 0)
# removing values < 0
dataset = dataset.drop(dataset[dataset['fare_amount'] < 0].index, axis = 0)
dataset['fare_amount'].sort_values(ascending = True)
# Removing fare_amount = 0
dataset = dataset.drop(dataset[dataset['fare_amount'] == 0].index, axis = 0)


# 2. Passenger_count
dataset['passenger_count'].describe()
dataset['passenger_count'].sort_values(ascending = False)
# Some Very High values in this variable as well
# removing values > 6
dataset = dataset.drop(dataset[dataset['passenger_count'] > 6].index, axis = 0)
# removing passenger_count = 0
dataset = dataset.drop(dataset[dataset['passenger_count'] == 0].index, axis = 0)
dataset['passenger_count'].sort_values(ascending = True)
# removing passenger_count = 0.12
dataset = dataset.drop(dataset[dataset['passenger_count'] == 0.12].index, axis = 0)

# Extracting Meaningful information(Distance) from coordinates.
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))

dataset['distance_travelled'] = distance(dataset['pickup_latitude'], dataset['pickup_longitude'],
                          dataset['dropoff_latitude'], dataset['dropoff_longitude'])
# Outlier Analysis For Ditance_travelled
dataset['distance_travelled'].sort_values(ascending = False)
dataset['distance_travelled'].describe()
# Removing values more than 1000kms
dataset = dataset.drop(dataset[dataset['distance_travelled'] > 1000].index, axis = 0)
dataset['distance_travelled'].describe()
# Now max distance is 129.95
from collections import Counter 
Counter(dataset['distance_travelled'] == 0)
Counter(dataset['fare_amount'] == 0)
Counter((dataset['distance_travelled'] == 0) & (dataset['fare_amount'] > 2.5))
Counter((dataset['distance_travelled'] == 0) & (dataset['fare_amount'] == 2.5))
# Removing distance_travelled = 0 from dataset where fare is > 2.5
dataset = dataset.drop(dataset[(dataset['distance_travelled'] == 0) & (dataset['fare_amount'] > 2.5)].index, axis = 0)

# Removing the pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude from dataset
dataset = dataset.iloc[:, [0,5,6,7,8,9,10,11]]


# Feature Selection
# Correlation Analysis(between numeric variables)
subset = dataset.iloc[:, [2,4,5,6,7]]

f, ax = plt.subplots(figsize = (7, 5))
corr = subset.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
           square = True, ax = ax)


# Visualisation
# Scatter Plot visualisation for different variables

# 1. Passenger_Count
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['passenger_count'], y = dataset['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()
# 2. Pickup_date
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['pickup_date'], y = dataset['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()
# 3. Pickup_day
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['pickup_day'], y = dataset['fare_amount'], s=10)
plt.xlabel('Pickup Day')
plt.ylabel('Fare')
plt.show()
# 4. Pickup_month
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['pickup_month'], y = dataset['fare_amount'], s=10)
plt.xlabel('Pickup Month')
plt.ylabel('Fare')
plt.show()
# 5. Pickup_hour
plt.figure(figsize=(15,7))
dataset.groupby(dataset["pickup_hour"])['pickup_hour'].count().plot(kind="bar")
plt.show()

plt.figure(figsize=(14,7))
plt.scatter(x = dataset['pickup_hour'], y = dataset['fare_amount'], s=10)
plt.xlabel('Pickup Hour')
plt.ylabel('Fare')
plt.show()
# 6. Pickup_minute
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['pickup_minute'], y = dataset['fare_amount'], s=10)
plt.xlabel('Pickup Minute')
plt.ylabel('Fare')
plt.show()
# 7. Distance_travelled
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['distance_travelled'], y = dataset['fare_amount'], s=10)
plt.xlabel('Distance Travelled')
plt.ylabel('Fare')
plt.show()

# Normality Check
for i in ['fare_amount']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()  
for i in ['distance_travelled']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['passenger_count']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['pickup_date']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['pickup_day']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['pickup_month']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['pickup_hour']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['pickup_minute']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
    
# Removing Skewness in distance_travelled and fare_amount
dataset['distance_travelled'] = np.log1p(dataset['distance_travelled'])
dataset['fare_amount'] = np.log1p(dataset['fare_amount'])
# Recheck for Normality
for i in ['fare_amount']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
for i in ['distance_travelled']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
    
# Seperating independent and dependent variables.
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

# SPLITTING THE DATA INTO TRAIN AND TEST.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Building Models

# 1. Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor_LR = LinearRegression()
regressor_LR.fit(X_train, Y_train)
# Predicting on Test Set
Y_pred = regressor_LR.predict(X_test)

# Calculating Mape
def MAPE(true, pred):
    mape = np.mean(np.abs((true - pred)/true))* 100
    return mape

MAPE(Y_test, Y_pred)

# RMSE
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# R Square

# Train Data
Y_pred_TR = regressor_LR.predict(X_train) 
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_TR)

# Test Data
r2_score(Y_test, Y_pred)

# 2. Decision Tree 
# Fitting Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(max_depth = 10, random_state = 0)
regressor_DT.fit(X_train, Y_train)

# Predicting on Test Set
Y_pred = regressor_DT.predict(X_test)

# Calculating Mape
MAPE(Y_test, Y_pred)

# RMSE
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# R Square
# Train Data
Y_pred_TR = regressor_DT.predict(X_train) 
r2_score(Y_train, Y_pred_TR)

# Test Data
r2_score(Y_test, Y_pred)

# 3. Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor(max_depth = 7, n_estimators = 300, random_state = 1)
regressor_RF.fit(X_train, Y_train)

# Predicting on Test Set
Y_pred = regressor_RF.predict(X_test)

# Calculating Mape
MAPE(Y_test, Y_pred)

# RMSE
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# R Square
# Train Data
Y_pred_TR = regressor_RF.predict(X_train) 
r2_score(Y_train, Y_pred_TR)

# Test Data
r2_score(Y_test, Y_pred)

# As Random Forest has the best values for RMSE and Rsquared hence 
# lets make it as the final model.
# Let's try some parameter tuning for RF.

# Applying K-Fold Cross Validation for Random Forest
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor_RF,
                             X = X_train,
                             y = Y_train,
                             cv = 7)
# calculating the mean of obtained accuracies.
accuracies.mean()

"""# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth' : [5,7,9], 'n_estimators' : [300, 400, 500],
               'random_state' : [0,1,2]}]
grid_search = GridSearchCV(estimator = regressor_RF,
                           param_grid = parameters,
                           cv = 7,
                           n_jobs = -1)
grid_search = grid_search.fit(X, Y)

# getting the scores and parameters
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_"""
# Commenting the above code as it takes time to run.

# The grid search suggests n_estimators = 300, random_state = 1. Lets try this on our
# regressor_RF and recheck the results.

# Finalising RandomForest Model for cab fare prediction on our actual Test
# Set.

# Reading and preparing the test set data for our model

dataset_test = pd.read_csv('test.csv')
# Converting to proper data types
dataset_test['pickup_datetime'] = pd.to_datetime(dataset_test['pickup_datetime'], errors = 'coerce')
dataset_test['passenger_count'] = dataset_test['passenger_count'].astype('object')
# Extracting useful information from 'pickup_datetime'
dataset_test['pickup_date'] = dataset_test['pickup_datetime'].dt.day
dataset_test['pickup_day'] = dataset_test['pickup_datetime'].dt.weekday
dataset_test['pickup_month'] = dataset_test['pickup_datetime'].dt.month
dataset_test['pickup_hour'] = dataset_test['pickup_datetime'].dt.hour
dataset_test['pickup_minute'] = dataset_test['pickup_datetime'].dt.minute
dataset_test['pickup_day'] = dataset_test['pickup_day'].astype('object')
dataset_test['pickup_hour'] = dataset_test['pickup_hour'].replace(0, 24)

# Removing pickup_datetime variable from dataset
dataset_test = dataset_test.iloc[:, [1,2,3,4,5,6,7,8,9,10]]

# Inspecting the Latitudes and longitudes
# Latitude(-90 - 90)
dataset_test[dataset_test['pickup_latitude'] < -90]
dataset_test[dataset_test['pickup_latitude'] > 90]

dataset_test[dataset_test['dropoff_latitude'] < -90]
dataset_test[dataset_test['dropoff_latitude'] > 90]

# Longitude(-180 - 180)
dataset_test[dataset_test['pickup_longitude'] < -180]
dataset_test[dataset_test['pickup_longitude'] > 180]

dataset_test[dataset_test['dropoff_longitude'] < -180]
dataset_test[dataset_test['dropoff_longitude'] > 180]
# None are beyond the range.

# 1. Missing Value Analysis.
dataset_test.isnull().sum()
# None Found

# 2. Exploring the variables
# 1. Passenger_count
dataset_test['passenger_count'].describe()
dataset_test['passenger_count'].sort_values(ascending = False)
dataset_test['passenger_count'].sort_values(ascending = True)

# Extracting Meaningful information(Distance) from coordinates.
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))

dataset_test['distance_travelled'] = distance(dataset_test['pickup_latitude'], dataset_test['pickup_longitude'],
                          dataset_test['dropoff_latitude'], dataset_test['dropoff_longitude'])
# Outlier Analysis For Ditance_travelled
dataset_test['distance_travelled'].sort_values(ascending = False)
dataset_test['distance_travelled'].describe()
# None found

# Removing the pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude from dataset
dataset_test = dataset_test.iloc[:, 4:11]

# Visualisations
# Normality Check
for i in ['distance_travelled']:
    print(i)
    sns.distplot(dataset_test[i],bins='auto',color='blue')
    plt.title("Distribution for Test Dataset Variable "+i)
    plt.ylabel("Density")
    plt.show()
# Removing Skewness in distance_travelled
dataset_test['distance_travelled'] = np.log1p(dataset_test['distance_travelled'])
# Normality ReCheck
for i in ['distance_travelled']:
    print(i)
    sns.distplot(dataset_test[i],bins='auto',color='blue')
    plt.title("Distribution for Test Dataset Variable "+i)
    plt.ylabel("Density")
    plt.show()

# Now building the final model with whole training data
from sklearn.ensemble import RandomForestRegressor
regressor_RF_final = RandomForestRegressor(max_depth = 7, n_estimators = 300, random_state = 1)
regressor_RF_final.fit(X, Y)

Y_pred_final = regressor_RF_final.predict(X) 
# Calculating Mape(Training Data)
MAPE(Y, Y_pred_final)
# MAPE obtained 8.41%, Accuracy = 92%

# RMSE(Training Data)
print(np.sqrt(metrics.mean_squared_error(Y, Y_pred_final)))

# R Square(Training Data)
r2_score(Y, Y_pred_final)

# Predicting on Test Set
predicted_fares = regressor_RF_final.predict(dataset_test)

"""# storing the predicted fares.
Predicted_fares = pd.DataFrame({'Predicted_Fares': predicted_fares})
Predicted_fares.to_csv('Final Predicted Fares.csv', index = False)"""

# Deleting temp variables
del i
del X_test
del X_train
del Y_pred
del Y_pred_TR
del Y_test
del Y_train
del accuracies
del subset

############################# END #######################################