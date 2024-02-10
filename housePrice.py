import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California Housing Prices dataset
california_housing = fetch_california_housing()

# Extract a single feature for simplicity (you can use the entire dataset if needed)
house_x = california_housing.data[:, np.newaxis, 5]  # For example, using the average rooms per dwelling
# house_x = california_housing.data  # Uncomment this line to use all features

# Split the data into training and testing sets
house_x_train, house_x_test, house_y_train, house_y_test = train_test_split(
    house_x, california_housing.target, test_size=0.2, random_state=42
)

# Create a linear regression model
house_model = LinearRegression()

# Train the model
house_model.fit(house_x_train, house_y_train)

# Make predictions on the test set
house_y_predicted = house_model.predict(house_x_test)

# Calculate and print the mean squared error
print("Mean Squared Error: ", mean_squared_error(house_y_test, house_y_predicted))

# Print the coefficients (weights) and intercept
print("Coefficients (Weights): ", house_model.coef_)
print("Intercept: ", house_model.intercept_)

# Plot the scatter plot and regression line
plt.scatter(house_x_test, house_y_test, color='black')
plt.plot(house_x_test, house_y_predicted, color='blue', linewidth=3)

# Set labels and display the plot
plt.xlabel("Average Rooms per Dwelling")
plt.ylabel("House Price")
plt.title("Linear Regression for House Price Prediction")
plt.show()

