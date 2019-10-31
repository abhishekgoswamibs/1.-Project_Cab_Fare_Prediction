# Cab Fare Prediction

setwd('D:/Data Science Edwisor/6. Projects/1. Project One Cab Rental Company')
# loading the dataset.
dataset = read.csv('train_cab.csv', na.strings = c(" ","", "NA"))

str(dataset)

# Extracting Meaningful Information from pickup_datetime.
library(lubridate)
dataset$pickup_datetime = ymd_hms(dataset$pickup_datetime)
dataset$pickup_date = day(dataset$pickup_datetime)
dataset$pickup_day = wday(dataset$pickup_datetime, label = FALSE)
dataset$pickup_month = month(dataset$pickup_datetime)
dataset$pickup_hour = hour(dataset$pickup_datetime)
str(dataset)

# Missing Value Analysis
sum(is.na(dataset))
apply(dataset, 2, function(x) {sum(is.na(x))})
# dropping the missing values
library(DataCombine)
dataset = DropNA(dataset)

# Outlier Analysis
# 1. Inspecting the latitudes and longitudes.
dataset[dataset['pickup_latitude'] < -90,]
dataset[dataset['pickup_latitude'] > 90,]
# Droping the row no 5687
dataset[dataset['pickup_latitude'] > 90,] = NA
dataset = DropNA(dataset)
dataset[dataset['dropoff_latitude'] < -90,]
dataset[dataset['dropoff_latitude'] > 90,]
dataset[dataset['pickup_longitude'] < -180,]
dataset[dataset['pickup_longitude'] > 180,]
dataset[dataset['dropoff_longitude'] < -180,]
dataset[dataset['dropoff_longitude'] > 180,]

# 2. fare_amount
dataset$fare_amount = as.numeric(as.character(dataset$fare_amount)) #Conversion generates one NA for 430- value in fare_amount.
summary(dataset$fare_amount)
dataset[order(-dataset$fare_amount),]
# Removing values > 150
dataset = DropNA(dataset)
dataset[dataset['fare_amount'] > 150,] = NA
dataset = DropNA(dataset)
dataset[order(dataset$fare_amount),'fare_amount']
# Removing values < 0
dataset[dataset['fare_amount'] < 0,] = NA
dataset = DropNA(dataset)
summary(dataset$fare_amount)
# Removing fare_amount = 0
dataset[dataset['fare_amount'] == 0,] = NA
dataset = DropNA(dataset)

# 3. Passenger_count
summary(dataset$passenger_count)
dataset[order(-dataset$passenger_count),'passenger_count']
# Removing passenger_count > 6
dataset[dataset['passenger_count'] > 6,] = NA
dataset = DropNA(dataset)
# Removing Passenger_count = 0
dataset[dataset['passenger_count'] == 0,] = NA
dataset = DropNA(dataset)
# Removing Passenger_count = 0.12
dataset[dataset['passenger_count'] == 0.12,] = NA
dataset = DropNA(dataset)
# Removing Passenger_count = 1.3
dataset[dataset['passenger_count'] == 1.3,] = NA
dataset = DropNA(dataset)

# Defining Haversine Distance Function
distance = function(lat1, lon1, lat2, lon2){
      p = 0.017453292519943295 # Pi/180
      a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
      d = 12742 * sin(sqrt(a))
      return(d)
}

dataset$distance_travelled = distance(dataset$pickup_latitude, dataset$pickup_longitude,
                                      dataset$dropoff_latitude, dataset$dropoff_longitude)
# Outlier Analysis for distance_travelled
dataset[order(-dataset$distance_travelled),'distance_travelled']
# removing some very high value in distance_travelled. i.e. distance_travelled > 1000
dataset[dataset['distance_travelled'] > 1000,] = NA
dataset = DropNA(dataset)
sum(is.na(dataset$distance_travelled))
summary(dataset$distance_travelled)
# Now max distance is 129.946
nrow(dataset[dataset['distance_travelled'] == 0,])
nrow(dataset[dataset['fare_amount'] == 0,])
nrow(dataset[(dataset['distance_travelled'] == 0) & (dataset['fare_amount'] > 2.5),])
nrow(dataset[(dataset['distance_travelled'] == 0) & (dataset['fare_amount'] == 2.5),])
nrow(dataset[(dataset['distance_travelled'] == 0) & (dataset['fare_amount'] < 2.5),])
# removing distance_travelled = 0 and fare amount > 2.5(assuming cancallation charge is 2.5)
dataset[(dataset['distance_travelled'] == 0) & (dataset['fare_amount'] > 2.5),'distance_travelled'] = NA
sum(is.na(dataset$distance_travelled))
dataset = DropNA(dataset)

# Removing pickup_datetime, and variables with longitudes and latitudes.
dataset = subset(dataset, select = -c(pickup_datetime, pickup_longitude, pickup_latitude, 
                                     dropoff_longitude, dropoff_latitude))
# Visualisations
# Scatter Plot visualisation for different variables.
library(ggplot2)

# 1. Passenger_count
ggplot() + 
  geom_point(aes(x = dataset$passenger_count, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Passenger_Count') +
  xlab('No of Passengers') +
  ylab('Fare')

# 2. Pickup_Date
ggplot() + 
  geom_point(aes(x = dataset$pickup_date, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Pickup_Date') +
  xlab('Date of Travel') +
  ylab('Fare')

# 3. Pickup_day
ggplot() + 
  geom_point(aes(x = dataset$pickup_day, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Pickup_Day') +
  xlab('Day of Travel') +
  ylab('Fare')

# 4. Pickup_month
ggplot() + 
  geom_point(aes(x = dataset$pickup_month, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Pickup_Month') +
  xlab('Month of Travel') +
  ylab('Fare')

# 5. Pickup_Hour
ggplot() + 
  geom_point(aes(x = dataset$pickup_hour, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Pickup_Hour') +
  xlab('Hour of Travel') +
  ylab('Fare')

# 6. Distance_Travelled
ggplot() + 
  geom_point(aes(x = dataset$distance_travelled, y = dataset$fare_amount),
             colour = 'red') +
  ggtitle(' Fare Vs Distance_Travelled') +
  xlab('Distacne Travelled') +
  ylab('Fare')

# Checking Distribution
# 1. fare_Amount
plot(density(dataset$fare_amount))
# we find that our data is skewed.
# Applying log to remove skewness.
dataset$fare_amount = log1p(dataset$fare_amount)
# Recheck
plot(density(dataset$fare_amount))

# 2. passenger_Count
plot(density(dataset$passenger_count))

# 3. pickup_date
plot(density(dataset$pickup_date))

# 4. pickup_day
plot(density(dataset$pickup_day))

# 5. pickup_month
plot(density(dataset$pickup_month))

# 6. pickup_hour
plot(density(dataset$pickup_hour))

# 7. distance_travelled
plot(density(dataset$distance_travelled))
# We see skewness in this data
# Removing skewness
dataset$distance_travelled = log1p(dataset$distance_travelled)
# Recheck
plot(density(dataset$distance_travelled))

# Converting the data into proper datatypes
dataset$passenger_count = as.factor(dataset$passenger_count)
dataset$pickup_day = as.factor(dataset$pickup_day)
dataset$pickup_hour = as.factor(dataset$pickup_hour)

# Splitting the data into train and test
library(caTools)

set.seed(123)
split = sample.split(dataset$fare_amount, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Building Models

# 1. Multiple Linear Regression Model
regressor_LR = lm(formula = fare_amount ~ .,
               data = training_set)
summary(regressor_LR)

# Predicting the test set
y_pred = predict(regressor_LR, newdata = test_set)

# MAPE CHECk
MAPE = function(y, x){
  mean(abs((y - x)/y)) * 100
}

MAPE(test_set[, 1], y_pred)

# Alternate method
library(DMwR)
regr.eval(test_set[,1], y_pred, stats = c('rmse', 'mape'))

# 2. Decision Tree Model
library(rpart)
regressor_DT = rpart(formula = fare_amount ~ .,
                     data = training_set,
                     control = rpart.control(minsplit = 7))
summary(regressor_DT)

# Predicting the test set
y_pred = predict(regressor_DT, newdata = test_set)

# Evaluating Performance
regr.eval(test_set[,1], y_pred, stats = c('rmse', 'mape'))

# 3. Random Forest Model
# Fitting Random Forest Regression Model to the model
library(randomForest)
set.seed(1234)
regressor_RF = randomForest(x = training_set[, 2:7],
                         y = training_set$fare_amount,
                         ntree = 500)
# Predicting the test set
y_pred = predict(regressor_RF, newdata = test_set)
# Evaluating Performance
regr.eval(test_set[,1], y_pred, stats = c('rmse', 'mape'))

# Finalising Random Forest model for our Cab Fare Prediction based on better rmse value
# and mape value is also very close to that of linear regression model.

# Reading and preparing the test set
dataset_test = read.csv('test.csv', na.strings = c(" ","", "NA"))

# Taking care of pickup_datetime variable.
dataset_test$pickup_datetime = ymd_hms(dataset_test$pickup_datetime)
dataset_test$pickup_date = day(dataset_test$pickup_datetime)
dataset_test$pickup_day = wday(dataset_test$pickup_datetime, label = FALSE)
dataset_test$pickup_month = month(dataset_test$pickup_datetime)
dataset_test$pickup_hour = hour(dataset_test$pickup_datetime)
str(dataset_test)

# Missing Value Analysis
sum(is.na(dataset_test))

# Outlier Analysis
# 1. Inspecting the latitudes and longitudes.
dataset_test[dataset_test['pickup_latitude'] < -90,]
dataset_test[dataset_test['pickup_latitude'] > 90,]
dataset_test[dataset_test['dropoff_latitude'] < -90,]
dataset_test[dataset_test['dropoff_latitude'] > 90,]
dataset_test[dataset_test['pickup_longitude'] < -180,]
dataset_test[dataset_test['pickup_longitude'] > 180,]
dataset_test[dataset_test['dropoff_longitude'] < -180,]
dataset_test[dataset_test['dropoff_longitude'] > 180,]

# 2. Passenger_count
summary(dataset_test$passenger_count)
dataset_test[order(-dataset_test$passenger_count),'passenger_count']
# None found

# Defining Haversine Distance Function
distance = function(lat1, lon1, lat2, lon2){
  p = 0.017453292519943295 # Pi/180
  a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
  d = 12742 * sin(sqrt(a))
  return(d)
}

dataset_test$distance_travelled = distance(dataset_test$pickup_latitude, dataset_test$pickup_longitude,
                                      dataset_test$dropoff_latitude, dataset_test$dropoff_longitude)
# Outlier Analysis for distance_travelled
dataset_test[order(-dataset_test$distance_travelled),'distance_travelled']
summary(dataset_test$distance_travelled)

# Removing pickup_datetime, and variables with longitudes and latitudes.
dataset_test = subset(dataset_test, select = -c(pickup_datetime, pickup_longitude, pickup_latitude, 
                                      dropoff_longitude, dropoff_latitude))
# Normality Check
# 1. distance_travelled
plot(density(dataset_test$distance_travelled))
# We see skewness in this data
# Removing skewness
dataset_test$distance_travelled = log1p(dataset_test$distance_travelled)
# Recheck
plot(density(dataset_test$distance_travelled))
# # 2. passenger_Count
# plot(density(dataset_test$passenger_count))
# # 3. pickup_date
# plot(density(dataset_test$pickup_date))
# # 4. pickup_day
# plot(density(dataset_test$pickup_day))
# # 5. pickup_month
# plot(density(dataset_test$pickup_month))
# # 6. pickup_hour
# plot(density(dataset_test$pickup_hour))
# Except distance_travelled none other variables are skewed

# Converting the data into proper datatypes
dataset_test$passenger_count = as.factor(dataset_test$passenger_count)
dataset_test$pickup_day = as.factor(dataset_test$pickup_day)
dataset_test$pickup_hour = as.factor(dataset_test$pickup_hour)

# 1. Final Random Forest Model
# Training Random Forest Regression Model to the whole training set
library(randomForest)
set.seed(1234)
regressor_RF_final = randomForest(x = dataset[, 2:7],
                            y = dataset$fare_amount,
                            ntree = 500)
y_pred_rf = predict(regressor_RF_final, newdata = dataset)
# Evaluating Performance
regr.eval(dataset[,1], y_pred_rf, stats = c('rmse', 'mape'))

# Making Final Predictions on our test set
y_pred_final = predict(regressor_RF_final, newdata = dataset_test)

# # Storing back the output
# cab_fare_prediction = data.frame(Predicted_Fares = y_pred_final)
# write.csv(cab_fare_prediction, 'cab fare predictions R.csv')

############################### END #####################################################