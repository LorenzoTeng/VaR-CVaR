import numpy as np
import math as m

class SGLD:

    def __init__(self, theta0, beta, gamma, lamda, q):
        self.theta0 = theta0
        self.beta = beta
        self.gamma = gamma
        self.lamda = lamda
        self.q = q

    def H0(self, x, theta):
        """ m = 1"""
        h = -self.q/(1-self.q) + (1/(1-self.q))*(x < theta) + 2 * self.gamma * theta
        return h

    def estimate(self, x_array):
        """x_array: numpy array sampled from distribution"""
        theta = self.theta0
        for i in range(len(x_array)):
            theta += -self.lamda * self.H0(x_array[i], theta) \
                     + m.sqrt((2*self.lamda/self.beta)) * np.random.normal(0, 1)

        expectation = self.gamma * theta**2
        for j in range(len(x_array)):
            expectation += ( theta + (1/(1-self.q)) * (x_array[j] - theta) * (x_array[j] > theta) )/len(x_array)

        return theta, expectation


the0 = 3
bet = 1e+8
gam = 1e-6
lam = 1e-4

Mdoel095 = SGLD(the0, bet, gam, lam, 0.95)
Mdoel099 = SGLD(the0, bet, gam, lam, 0.99)

xmu0sig1 = np.random.normal(0, 1, int(1e+6))
xmu1sig2 = np.random.normal(1, 2, int(1e+6))
xmu3sig5 = np.random.normal(3, 5, int(1e+6))

xt10 = np.random.standard_t(10, int(1e+6))
xt7 = np.random.standard_t(7, int(1e+6))
xt3 = np.random.standard_t(3, int(1e+6))

print(Mdoel095.estimate(xmu0sig1))
print(Mdoel095.estimate(xmu1sig2))
print(Mdoel095.estimate(xmu3sig5))

print(Mdoel095.estimate(xt10))
print(Mdoel095.estimate(xt7))
print(Mdoel095.estimate(xt3))

print(Mdoel099.estimate(xmu0sig1))
print(Mdoel099.estimate(xmu1sig2))
print(Mdoel099.estimate(xmu3sig5))

print(Mdoel099.estimate(xt10))
print(Mdoel099.estimate(xt7))
print(Mdoel099.estimate(xt3))