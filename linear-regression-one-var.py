"""
    Problem:    Predict profit of a food truck
    Data:       profit_data.txt
    Algorithm:  Univariate linear regression
"""

import sys


def h(theta, x):
    return theta[0] * x[0] + theta[1] * x[1]


def cost(theta, x, y, m):
    summation = sum([(h(theta, x[i]) - y[i]) ** 2
                        for i in xrange(m)])
    return summation / (2 * m)


def gradientdescent(theta, x, y, m, alpha, iterations):
    for i in xrange(iterations):
        thetatemp = theta[:]
        for j in range(2):
            summation = sum([(h(theta, x[k]) - y[k]) * x[k][j]
                                for k in xrange(m)])
            thetatemp[j] = thetatemp[j] - alpha * summation / m
        theta = thetatemp[:]
    return theta


def main():
    x = []  # List of training example parameters
    y = []  # List of trainign example results

    for line in sys.stdin:
        [a, b] = map(float, line.split(','))
        x.append(a)
        y.append(b)

    m = len(x)  # Number of training examples

    x = [[1, i] for i in x]  # Append a column of 1's to x

    # Initialize theta's
    initialtheta = [0.0, 0.0]
    learningrate = 0.01
    iterations = 10000

    # Run gradient descent to get our guessed hypothesis
    finaltheta = gradientdescent(initialtheta,
                                 x, y, m,
                                 learningrate, iterations)

    # Evaluate our hypothesis accuracy
    print "Initial cost:", cost(initialtheta, x, y, m)
    print "Final cost:", cost(finaltheta, x, y, m)

if __name__ == "__main__":
    main()
