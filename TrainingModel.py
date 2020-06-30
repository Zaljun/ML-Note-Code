# LINEAR REGRESSION

# NORMAL EQUATION
# Generate linear-looking data to test equation
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1) 
# y = 4 + 3x1 + Gaussian noise

from matplotlib import pyplot as plt
plt.title("Linear Reg")
plt.xlabel("X1")
plt.ylabel("y")
#plt.plot(X, y, "b.")
plt.scatter(X, y)

# Compute theta hat using NORMAL EQUATION
X_b = np.c_[np.ones((100,1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best

X_new = np.array([[0],[2]])
X_new_b =np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# USING sklearn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond = 1e-6)
theta_best_svd

np.linalg.pinv(X_b).dot(y)
'''
The pseudoinverse itself is computed using a standard matrix factorization
technique called Singular Value Decomposition (SVD) that can decompose the
training set matrix X into the matrix multiplication of three matrices U Σ V(T)
(see numpy.linalg.svd()).
'''
# GRADIENT DESCENT
# Batch GD (Full GD)
eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

theta

# Stochastic GD 
# (Using A SIMPLE LEARNING SCHEDULE)
n_epochs = 50
t0, t1 = 5, 50 #learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.rand(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
theta

# USING sklearn
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 1000, tol = 1e-3, penalty = None, eta0 = 0.1)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_

# Mini-batch GD


# POLYNOMIAL REGRESSION
# Generate a nonlienar looking dataset based on a quadratic equation
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(m, 1)

plt.plot(X, y, "b.")

# USING sklearn's PolynomialFeatures class to transform training data, 
# adding the square (second-degree) of each feature in the set as a new feature 
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
X[0], X_poly[0]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

# LEARNING CURVES
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size = 0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "val")
    #plt.axes([0,80,0,5])
    plt.show()

# Linear Model
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# 10 Degree Plynomial Model
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 10, include_bias = False)),
        ("lin_reg", LinearRegression()),
        ])
    
plot_learning_curves(polynomial_regression, X, y)

# REGULARIZED LINEAR MODELS
# RIDGE REGRESSION

