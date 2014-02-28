import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

def regress_theta(X, y, Lambda):
    """
      Computes the theta that minimizes
      (1/(2n))||y - X*theta||^2 + 1/2 lambda||theta||^2
    """
    lsreg = linear_model.Ridge(fit_intercept=False);
    lsreg.set_params(alpha=Lambda);
    lsreg.fit(X,y);
    return lsreg.coef_;

def noisy_quad_fit(order, Lambda, n_train=20, n_test=81):
    """
      Creates n_train training data points with noise, fits to poly of order,
      then tests on n_test points (noise free).  Uses offset quadratic
    """
    low_x = -2;
    high_x = 2;
    plt.close('all');

    train_x = np.linspace(low_x, high_x, n_train);
    X = np.vander(train_x, N = order+1);
    y = (1+train_x**2)# + 0.6*(np.random.rand(n_train) - 0.5);
    # y = np.sin(3 * train_x) - train_x * np.cos(2 * train_x);# + 0.6*(np.random.rand(n_train) - 0.5);
    theta = regress_theta(X,y,Lambda);
    predict_y = np.dot(X,theta);
    print 'Training Error = ', np.max(np.abs(y - predict_y));

    test_x = np.linspace(low_x, high_x, n_test);
    Xt = np.vander(test_x, N = order+1);
    yt = 1+test_x**2;
    # yt = np.sin(3 * test_x) - test_x * np.cos(2 * test_x)
    predict_yt = np.dot(Xt,theta);
    print 'Testing Error = ', np.max(np.abs(yt - predict_yt));

    # plt.plot(train_x, y, 'ro');
    # plt.plot(train_x, predict_y, 'rx');
    # plt.plot(test_x, predict_yt, 'bx');
    # plt.show();
    return (np.max(np.abs(y - predict_y)), np.max(np.abs(yt - predict_yt)))

trainingError = 1
minTrainI = 0
minTrainJ = 0
testingError = 1
minTestI = 0
minTestJ = 0

lambdas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
for i in range(2,19):
    for j, l in enumerate(lambdas):
        trainError, testError = noisy_quad_fit(i, l)
        if trainError < trainingError:
            trainingError = trainError
            minTrainI = i
            minTrainJ  = j
        if testError < testingError:
            testingError = testError
            minTestI = i
            minTestJ = j

print "Minimum training error:", trainingError, "at i=", minTrainI, "j=", minTrainJ, lambdas[minTrainJ]
print "Minimum testing error:", testingError, "at i=", minTestI, "j=", minTestJ, lambdas[minTestJ]

# noisy_quad_fit(17, 0.000001)
# noisy_quad_fit(0, 0.01)
