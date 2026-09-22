"""
import random
with open("data_linear3.csv", "w") as file:
    file.write("x,y\n")
    m = -2.1274 #+ random.random()
    c = 4.5679 #random.choice([_ for _ in range(-1000, 1001)])
    for x in range(0,1001):
        X = x
        noise = random.gauss(0, 700)
        Y = m * X + c + noise
        file.write(f"{X},{Y}\n")
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def mse(predicted_m, predicted_c, actual_y_values, x_values):
    """
## Brief overview of this function:
Calculates the Mean Squared Error (MSE).

Args:

    predicted_m (float):the predicted m value/gradient.

    predicted_c (float):the predicted c value/y-intercept.

    actual_y_values (list or numpy.ndarray):a list or array of the y-values/y-coordinates from the dataset.

    x_values (list or numpy.ndarray):a list or array of the x-values/x-coordinates from the dataset. 
    
Returns:
    float:
    A value with a floating/decimal point that is the mean squared error
    """
    total_squared_error = 0
    n = 0
    for x, y in zip(x_values, actual_y_values):
        n += 1
        total_squared_error += (y - (predicted_m * x + predicted_c)) ** 2
    mean_squared_error = total_squared_error / n
    return mean_squared_error


def de_by_dm(actual_y_values, x_values, m_now, c_now):
    """
## Brief overview of this function:
Calculates the partial derivative of error/loss function with respect to m/gradient.

Args:

    actual_y_values (list or numpy.ndarray):a list or array of the y-values/y-coordinates from the dataset.

    x_values (list or numpy.ndarray):a list or array of the x-values/x-coordinates from the dataset.

    m_now (float):m_now is basically the current m (gradient) value.

    c_now (float):c_now is basically the current c (y-intercept) value.
    
Returns:
    float:
    A value with floating/decimal point which is the partial derivative of error/loss function with 
    respect to m/gradient.

    """
    pd_of_e_by_m = 0
    n = 0
    # for x in x_values:
    #    n += 1
    #    pd_of_e_by_m += x * (next(actual_y_values) - (m_now * x + c_now))
    for x, y in zip(x_values, actual_y_values):
        n += 1
        pd_of_e_by_m += x * (y - (m_now * x + c_now))

    pd_of_e_by_m = (-2 / n) * pd_of_e_by_m
    return pd_of_e_by_m


def de_by_dc(actual_y_values, x_values, m_now, c_now):
    """
## Brief overview of this function:
Calculates the partial derivative of error/loss function with respect to c/y-intercept.

Args:

    actual_y_values (list or numpy.ndarray):a list or array of the y-values/y-coordinates from the dataset.

    x_values (list or numpy.ndarray):a list or array of the x-values/x-coordinates from the dataset.

    m_now (float):m_now is basically the current m (gradient) value.

    c_now (float):c_now is basically the current c (y-intercept) value.
    
Returns:
    float:
    A value with floating/decimal point which is the partial derivative of error/loss function with 
    respect to c/y-intercept.

    """
    pd_of_e_by_c = 0
    n = 0
    # for x in x_values:
    #    n += 1
    #    pd_of_e_by_c += next(actual_y_values) - (m_now * x + c_now)

    for x, y in zip(x_values, actual_y_values):
        n += 1
        pd_of_e_by_c += y - (m_now * x + c_now)

    pd_of_e_by_c = (-2 / n) * pd_of_e_by_c
    return pd_of_e_by_c

def adagrad(initial_learning_rate, current_derivative, squared_sum_of_derivatives):
    """
## Brief overview of this function
Finds the new, optimal learning rate using the Adagrad optimizer algorithm to help the model
reach convergence faster.

Args:
    initial_learning_rate (float): the learning rate with which the linear regression model started training.
        
    current_derivative (float): the partial derivative of error/loss with respect to m/gradient or c/y-intercept.

    squared_sum_of_derivatives (float): the squared sum of the partial derivative of error/loss 
                                        with respect to m/gradient or c/y-intercept.

Returns:
    float:
    a value with a floating/decimal point that is the new, optimal learning rate that will help the model reach
    convergence faster.
    """
    squared_sum_of_derivatives = squared_sum_of_derivatives + (current_derivative ** 2)
    new_learning_rate = initial_learning_rate / ((squared_sum_of_derivatives + (1e-8)) ** 0.5)
    return (new_learning_rate, squared_sum_of_derivatives)

def rmsprop(initial_learning_rate, current_derivative, weighted_squared_sum_of_derivatives, beta):
    """
## Brief overview of the function
Finds the new, optimal learning rate using RMSProp (Root Mean Squared Propagation) optimizer algorithm to help the 
model reach covergence faster.

Args:
    initial_learning_rate (float): the learning rate with which the linear regression model started training.
        
    current_derivative (float): the partial derivative of error/loss with respect to m/gradient or c/y-intercept.

    weighted_squared_sum_of_derivatives (float): the weighted squared sum of the partial derivative of error/loss 
                                        with respect to m/gradient or c/y-intercept.
    
    beta (float): the decay factor by which the weighted squared sum of the partial derivative and current partial 
                derivative is multiplied by.

Returns:
    tuple:
        A tuple containing 
        1) Value with a floating/decimal point, which is the new, optimal learning rate that will help the model reach
           convergence faster.
        2) Weighted squared sum of derivatives
    
    """
    weighted_squared_sum_of_derivatives = (beta * weighted_squared_sum_of_derivatives) + (1-beta) * (current_derivative ** 2)
    new_learning_rate = initial_learning_rate / ((weighted_squared_sum_of_derivatives + (1e-8)) ** 0.5)
    return (new_learning_rate, weighted_squared_sum_of_derivatives)

def adam():
    pass

def gradient_descent(m_now, c_now, actual_y_values, x_values, initial_learning_rate, optimizer=None, **kwargs):
    """
## Brief overview of this function:
Performs gradient descent once (when called).

Args:

    m_now (float):m_now is basically the current m (gradient) value.

    c_now (float):c_now is basically the current c (y-intercept) value.

    actual_y_values (list or numpy.ndarray):a list or array of the y-values/y-coordinates from the dataset.

    x_values (list or numpy.ndarray):a list or array of the x-values/x-coordinates from the dataset.

    initial_learning_rate (float):a learning rate that will be used initially (if optimizer is enabled) 
        or will always be used, i.e, the value of learning rate will not change/static learning rate
        (if optimizer is disabled).

    optimizer (str, optional):this actually is the algorithm used to change learning rate dynamically so that
        convergence is reached quickly. 
        Default value is None, which means the function uses a static learning rate. 
        Other values that are allowed are:
            - "adagrad"
            - "rmsprop"
            - "adam"     
    
        These values should be passed as strings and spelling must be the same as the ones mentioned above, 
        otherwise the function will simply use the default value (None).

Returns:
    tuple:
    A tuple containing values in this order
    1) partial derivative of error/loss function with respect to m/gradient 
    2) partial derivative of error/loss function with respect to c/y-intercept 
    3) updated value of m/gradient 
    4) updated value of c/y-intercept

    """
    # list_actual_y_values = list(actual_y_values)
    # list_x_values = list(x_values)

    # actual_y_values_for_de_by_dm = (ym for ym in list_actual_y_values)
    # actual_y_values_for_de_by_dc = (yc for yc in list_actual_y_values)

    # x_values_for_de_by_dm = (xm for xm in list_x_values)
    # x_values_for_de_by_dc = (xc for xc in list_x_values)

    # del list_actual_y_values
    # del list_x_values

    global squared_sum_of_pd_of_e_by_m, squared_sum_of_pd_of_e_by_c, beta, momentum_of_pd_of_e_by_m, momentum_of_pd_of_e_by_c

    pd_of_e_by_m = de_by_dm(actual_y_values, x_values, m_now, c_now)
    pd_of_e_by_c = de_by_dc(actual_y_values, x_values, m_now, c_now)

    optimizer = optimizer.lower() if isinstance(optimizer, str) else None

    if optimizer == "adagrad":
        learning_rate_for_pd_of_e_by_m, squared_sum_of_pd_of_e_by_m = adagrad(initial_learning_rate, pd_of_e_by_m, squared_sum_of_pd_of_e_by_m)
        learning_rate_for_pd_of_e_by_c, squared_sum_of_pd_of_e_by_c = adagrad(initial_learning_rate, pd_of_e_by_c, squared_sum_of_pd_of_e_by_c)
        m_now = m_now - (learning_rate_for_pd_of_e_by_m * pd_of_e_by_m)
        c_now = c_now - (learning_rate_for_pd_of_e_by_c * pd_of_e_by_c)
    
    elif optimizer == "rmsprop":
        pass
    elif optimizer == "adam":
        pass
    else:
        m_now = m_now - (initial_learning_rate * pd_of_e_by_m)
        c_now = c_now - (initial_learning_rate * pd_of_e_by_c)
    
    return (pd_of_e_by_m, pd_of_e_by_c, m_now, c_now,)

def standardise_data(og_y_values, og_x_values):
    """
## Brief overview of this function:
Standardises the dataset (by using Z-score normalisation) so that the x and y values have the same scale but 
the features of the dataset, such as the variation in the data, are still retained. This helps the model capture 
the pattern of the data more effectively instead of just memorising the data.

Args:
    og_y_values (list or numpy.ndarray): a list of original y-values.

    og_x_values (list or numpy.ndarray): a list of original x-values.

Returns:
    tuple:
    A tuple where 
    1. The first value is the standardised y-values (type is numpy.ndarray).
    2. The second value is the standardised x-values (type is numpy.ndarray). 
    3. The third value is the mean of original y-values (type is float).
    4. The fourth value is the mean of original x-values (type is float), 
    5. The fifth value is the standard deviation of the original y-values (type is float). 
    6. The sixth value is the standard deviation of the original x-values (type is float).
    """
    if isinstance(og_y_values, list):
        og_y_values = np.array(og_y_values)
    
    if isinstance(og_x_values, list):
        og_x_values = np.array(og_x_values)

    mean_of_y = og_y_values.mean()
    mean_of_x = og_x_values.mean()

    std_dev_of_y = og_y_values.std()
    std_dev_of_x = og_x_values.std()

    y_standardised = (og_y_values - mean_of_y) / std_dev_of_y
    x_standardised = (og_x_values - mean_of_x) / std_dev_of_x
    
    return (y_standardised, x_standardised, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x)

def unstandardise_m_and_c(m_standardised, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x):
    """
## Brief overview of this function
Converts the standardised m/gradient value and c/y-intercept value to their unstandardised value. The standardised 
m/gradient value and c/y-intercept value are the values output by the gradient descent function when the function
is used/run on standardised data, which is the correct value for the standadised data but not for the
unstandardised/original data. 
## Note: 
This function does not have a parameter/argument that accepts the
standardised c/y-intercept value. This is because, when the data is standardised, c/y-intercept value becomes zero.

Args:
    m_standardised (float): the m/gradient value generated by the gradient descent function when its run on
                            standardised data.
    
    mean_of_y (float): the mean of the original y-values/y-values straight from the dataset.

    mean_of_x (float): the mean of the original x-values/x-values straight from the dataset.

    std_dev_of_y (float): the standard deviation of the original y-values/y-values straight from the dataset.

    std_dev_of_x (float): the standard deviation of the original x-values/x-values straight from the dataset.

Returns:
    tuple:
    A tuple containing
    1. The unstandardised m/gradient value.
    2. The unstandardised c/y-intercept value.

    """
    m_unstandardised = (std_dev_of_y / std_dev_of_x) * m_standardised
    c_unstandardised = mean_of_y - (m_unstandardised * mean_of_x)
    return (m_unstandardised, c_unstandardised)


df = pd.read_csv("data_linear3.csv")

n = 1
current_m = 0
current_c = 0
learning_rate = 0.8
squared_sum_of_pd_of_e_by_m = 0
squared_sum_of_pd_of_e_by_c = 0
beta = 0
momentum_of_pd_of_e_by_m = 0
momentum_of_pd_of_e_by_c = 0

y = np.array(df["y"])
x = np.array(df["x"])

iteration = []
error = []

y_standardised, x_standardised, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x = standardise_data(y, x)

#plt.figure(1)
#plt.scatter(x, y)
#plt.figure(2)
#plt.scatter(x_standardised, y_standardised)
#plt.show()


while True:
    try:
        print(mse(current_m, current_c, y_standardised, x_standardised))
        pd_m, pd_c, current_m, current_c = gradient_descent(
            current_m, current_c, y_standardised, x_standardised, learning_rate, optimizer="adagrad"
        )
        
        iteration.append(n)
        error.append(mse(current_m, current_c, y_standardised, x_standardised))

        if math.isclose(pd_m, 0, abs_tol=1e-10) and math.isclose(pd_c, 0, abs_tol=1e-10):
            break
        else:
            n += 1

    except KeyboardInterrupt:
        break

#print(current_m, current_c)

#current_m = 2.276455841538912
#current_c = -624.9202229368345

actual_m, actual_c = unstandardise_m_and_c(current_m, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x)

print("Adagrad:")
print(actual_m, actual_c, n)

xpi = list(range(len(df["x"])))
ypi = [actual_m * x + actual_c for x in xpi]

"""
###
iteration2 = []
error2 = []

m = 0
current_m = 0
current_c = 0
learning_rate = 0.05

y = np.array(df["y"])
x = np.array(df["x"])


y_standardised, x_standardised, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x = standardise_data(y, x)


while True:
    #lr = learning_rate #/ (1 + decay_rate * n)
    iteration2.append(m)
    error2.append(mse(current_m, current_c, y_standardised, x_standardised))
    m += 1
    try:
        #print(mse(current_m, current_c, y_standardised, x_standardised))
        pd_m, pd_c, current_m, current_c = gradient_descent(
            current_m, current_c, y_standardised, x_standardised, learning_rate
        )
        if m == 200:#(math.isclose(pd_m, 0, abs_tol=1e-10) and math.isclose(pd_c, 0, abs_tol=1e-10)) or m == 10:
            break

    except KeyboardInterrupt:
        break


actual_m, actual_c = unstandardise_m_and_c(current_m, mean_of_y, mean_of_x, std_dev_of_y, std_dev_of_x)

print("Vanilla:")
print(actual_m, actual_c, m)

xpj = list(range(len(df["x"])))
ypj = [actual_m * x + actual_c for x in xpj]

###
"""




plt.figure(1)
plt.scatter(df["x"], df["y"])
plt.plot(xpi, ypi, color='green', label='Adagrad:')
"""
###
plt.plot(xpj, ypj, color='red', label='Vanilla:')
"""
plt.figure(2)
plt.plot(iteration, error, color='green', label='Adagrad:')
"""
###
plt.plot(iteration2, error2, color='red', label='Vanilla:')
"""
plt.legend()
plt.xlabel("Number of Iteration")
plt.ylabel("Error/MSE score")

plt.show()

"""
from sklearn.linear_model import LinearRegression
import numpy as np

# Convert x and y to numpy arrays and reshape x for sklearn
x_np = np.array(x).reshape(-1, 1)
y_np = np.array(y)

# Fit linear regression using sklearn
model = LinearRegression()
model.fit(x_np, y_np)


# Predict using sklearn model
y_pred = model.predict(x_np)


# Plot both your line (green) and sklearn's line (red)
plt.scatter(x, y, label='Data')
plt.plot(x, ypi, color='green', label='Your Gradient Descent Adagrad')

plt.plot(x, y_pred, color='red', linestyle='--', label='Sklearn LinearRegression')
plt.legend()
plt.title("Comparison: Your Line vs Sklearn Line")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
