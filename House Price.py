# importing Dependencies
import numpy as np  # for numpy arrays
import pandas as pd  # for reading data
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for plotting
from sklearn.datasets import fetch_california_housing  # house price dataset is extracted from this lib
from sklearn.model_selection import train_test_split  # for splitting the data
from xgboost import XGBRegressor  # we are using regression model for prediction
from sklearn import metrics

# Importing the House Price Dataset

house_price_dataset = fetch_california_housing()
# print(house_price_dataset)  # target is the label of the dataset

# To make the Dataset more structured we use the pandas library
# Loading the dataset into Pandas Dataframe
# We include the columns from the dataset as feature names. If we don't include it our data will be represented without column names
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# to Download a Dataset into a txt/.csv format.
# house_price_dataframe.to_csv('california_housing_price.csv', sep = ',', index = True)
# Read 1st five values of the dataset
# print(house_price_dataframe.head())

# Adding the target column to out dataframe
# we create a new column['price'] and add the target values to it
house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe)

# Now let's check the no. of rows and columns we have in the dataframe
# print(house_price_dataframe.shape)  # 20640 rows and 9 cols

# Now we need to check if our dataset has missing values
# we use the isnull() function to find them as we cannot feed null values to our model
# print(house_price_dataframe.isnull().sum())  # .sum()->counts no.of missing values

# we don't have missing values in our dataset. Now lets find the statistical measures of the data
# print(house_price_dataframe.describe())  # 25%,50%,75% percentile mean that there are 25% values are present than the given value in each feature

# Understanding Correlation between Features(Columns) in the dataset
# 1.Positive Correlation ->  When one feature value increases than the other Feature value also increases
# eg. if the rooms in house increases the price increases
# 2. Negative Correlation -> When value of One Feature increases the other decreases
# eg. if the crime in area increases the house price decreases

correlation = (house_price_dataframe.corr())

# Constructing a heatmap to understand correlation
plt.figure(figsize=(10, 10))  # map size
# correlation, colorbar, values in square, 1f means 1 value after decimal point, annot means feature names and it's size 8, cmap is color of the haetmap
# print(sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues'))

# Splitting the labelled data and numerical values
# dropping the price col and storing rest of the col data into X
X = house_price_dataframe.drop(['price'], axis=1)
# storing labelled data 'price' into Y
Y = house_price_dataframe['price']
# print(X, Y)

# Splitting the data into Training Data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# Now lets. check the no.of rows and cols in X, X_train and X-test
# print(X.shape, X_train.shape, X_test.shape)

# Model Training (We are using XGBoost Regressor Model)
# loading the model
model = XGBRegressor()
# training the model
model.fit(X_train, Y_train)

# train_data_prediction = model.predict(X_train)
# print(train_data_prediction)

# Now we check if the prediction made by our model is correct to the original values
# R square error helps us to correct the error is our predicted values
# We use metrics.r2_score to find the variance between predicted values against original values
# train_error_1 = metrics.r2_score(Y_train, train_data_prediction)
# print('R Squared Error(train data):', train_error_1)  # if it is closer to 0 that means our model is working fine

# Mean Absolute error gives us mean value by subtracting the predicted value from original value
# train_error_2 = metrics.mean_absolute_error(Y_train, train_data_prediction)
# print('Mean Absolute Error(train data):', train_error_2)

# prediction of test data
# test_data_prediction = model.predict(X_test)
# test_error_1 = metrics.r2_score(Y_test, test_data_prediction)
# print('R Squared Error(test data):', test_error_1)

# test_error_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
# print('Mean Absolute Error(test data):', test_error_2)

# Visualizing the actual price and predicted price using graph
# plt.scatter(Y_train, train_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual Price VS Predicted Price")
# plt.show()

# 0R
# predicting by inputting feature values
input_data = (2.6736,52.0,4.0,1.0977011494252873,345.0,1.9827586206896552,37.84,-122.26)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)
