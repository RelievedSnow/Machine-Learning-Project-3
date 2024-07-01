# Machine-Learning-Project-3.
# Project 3: House Price Prediction (California_house_price).

# Step 1: Importing Dependencies.
* import numpy for numpy arrays.
* import pandas for reading data.
* import matplotlib library for plotting graphs.
* import seaborn library for plotiing more graphs.
* import sklearn.datasets for importing the california_house_price dataset.
* import sklearn.model_selection for splitting training and test data.
* import XGBRegressor Model for prediction from xgboost library as we are using regression model for prediction.
* import metrics from sklearn

# Step 2: Data Collection and Data Pre-processing.
* Importing the House Price Dataset through fetch_california_housing() function.
* To make the Dataset more structured we use the pandas library.
* Loading the dataset into Pandas Dataframe through 'pd.DataFrame()' function.
* We include the columns from the dataset as feature names. If we don't include it our data will be represented without column names.
* Reading the 1st five values of the dataset using'.head()' function. As we can see we do not have the target column i.e. 'price'.
* Adding the target column to out dataframe.
* We create a new column['price'] and add the target values to it.
* Now let's check the no. of rows and columns we have in the dataframe using the '.shape' methode.
* Now we need to check if our dataset has missing values
* We use the 'isnull()' function to find them as we cannot feed null values to our model. The '.sum()' function counts all the null values.
* Now lets find the statistical measures of the data using '.describe()' function.

# Understanding Correlation between Features(Columns) in the dataset.
* 1.Positive Correlation ->  When one feature value increases than the other Feature value also increases.
* eg. if the rooms in house increases the price increases.
* 2. Negative Correlation -> When value of One Feature increases the other decreases.
* eg. if the crime in area increases the house price decreases.
* Constructing a heatmap to understand correlation.
* Correlation, colorbar, values in square, 1f means 1 value after decimal point, annot means feature names and it's size 8, cmap is color of the haetmap.

# Splitting the labelled data and numerical values.
* Dropping the price col and storing rest of the col data into X.
* Storing labelled data 'price' into Y.

# Splitting the data into Training Data and Test Data.
* We split the Data into Training Data and Testing Data using the train_test_split.
* 'test_size' is the percent in which our dataset in split for traning and testing. [0.1 = 10%-90%, 0.2 = 20%-80%].
* random_state() function is a pseudo number that allows you to reproduce the same train_test split each time.

# Model Training (We are using XGBoost Regressor Model).
* Loading the model by using XGBRegressor() and storing it's function in the 'model' variable. 
* Now we train the model using the model.fit(X_train, Y_train).
* We make Prediction using 'model.predict()' funtion and pass it the 'X_train' values.
* Now we check if the prediction made by our model is correct to the original values. To correct the errors in out predicted values we use two Methods.
1. R Square Error.
2. Mean Absolute Error.
* We use 'metrics.r2_score' to find the variance between predicted values against original values.
* Mean Absolute error gives us mean value by subtracting the predicted value from original value.
* We now perform the same steps on the Test data as our model is now familiar with train data but not Test data. Then we will check its R square error and Mean absolte error values and compare them.

# We can Predect the Data in two ways(Data without any Errors).
1. Visualizing the actual price and predicted price using graph(Scatter Graph).
* We use the .scatter() function and provide the 'Y_train' labelled data againt the predicted data and apply proper labels and Title to the graph.
2. Predicting by inputting feature values.
* We create a variable 'input_data' to enter the input values as list.
* We reshape the input data from 1-D to 2-D as we are providing it in 1-D form.
* We convert the 'input_data_reshaped' value into standardized data as we have provided the model with 'standardized_data'. So if we provide raw values it will fail to predict.
* Now we make predictions.
