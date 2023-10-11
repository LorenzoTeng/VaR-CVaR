import numpy as np
import math as m
import pandas as pd

class SGLD_portfolios:

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

    def cal_g(self, w):
        """
        w: numpy array column 3*1
        g: numpy array column 3*1
        partial_g: numpy matrix 3*3
        """
        g = np.zeros_like(w)
        partial_g = np.zeros([len(w),len(w)])

        for i in range(len(w)):
            g[i] = m.exp(w[i])

        denominator = sum(g)

        for j in range(len(w)):
            for k in range(len(w)):
                if j == k:
                    partial_g[j][k] = (g[j]*(denominator-g[j])) / (denominator**2)
                else:
                    partial_g[j][k] = -(g[j] * g[k]) / (denominator ** 2)

        g = g/denominator
        return g, partial_g

    def H(self, x, theta, w):
        """x: numpy array column 3*1"""
        g_w, partial_g_w = self.cal_g(w)
        indicator = (np.dot(g_w.squeeze(), x) >= theta)
        h_theta = 1 - (1/(1-self.q)) * indicator + 2 * self.gamma * theta
        h_omega = ((1/(1-self.q)) * indicator * np.dot(partial_g_w, x))[:,np.newaxis] \
                  + 2 * self.gamma * w

        return h_theta, h_omega

    def estimate(self, x_array):
        """x_array: numpy ndarray 3*n sampled from distribution of assests, each row represents an asset"""
        theta = self.theta0
        w = np.zeros([3,1])
        for i in range(x_array.shape[1]):
            h_theta, h_omega = self.H(x_array[:,i], theta, w)
            theta += -self.lamda * h_theta \
                     + m.sqrt((2*self.lamda/self.beta)) * np.random.normal(0, 1)
            w += -self.lamda * h_omega \
                     + m.sqrt((2*self.lamda/self.beta)) * np.random.normal(0, 1, 3)[:,np.newaxis]

        g_w_exp, blabla = self.cal_g(w)
        expectation = self.gamma * (theta**2 + np.linalg.norm(w, ord=2))
        for j in range(x_array.shape[1]):
            x = x_array[:,j]
            indicator = (np.dot(g_w_exp.squeeze(), x) >= theta)
            expectation += ( theta + (1/(1-self.q)) * (np.dot(g_w_exp.squeeze(), x) - theta) * indicator )/(x_array.shape[1])

        return np.hstack([g_w_exp.squeeze(), theta, expectation])


the0 = 3
bet = 1e+8
gam = 1e-8
lam = 1e-4

Model = SGLD_portfolios(the0, bet, gam, lam, 0.95)

""""Notice that in N(0, 10^6), sigma = 1e+3 and similar is others"""
# #case1
asset1 =        np.random.normal(500, 1,         int(1e+6))[np.newaxis,:]
asset2 =        np.random.normal(0,   int(1e+3), int(1e+6))[np.newaxis,:]
asset3 = 0.01 * np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]

case1 = np.concatenate((asset1, asset2, asset3), axis=0)
a = Model.estimate(case1)


# #case2
asset1 = np.random.normal(500, 1,         int(1e+6))[np.newaxis,:]
asset2 = np.random.normal(0,   int(1e+3), int(1e+6))[np.newaxis,:]
asset3 = np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]

case2 = np.concatenate((asset1, asset2, asset3), axis=0)

# #case3
asset1 = m.sqrt(1000) * np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]
asset2 =                np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]
asset3 =                np.random.normal(0,   2,         int(1e+6))[np.newaxis,:]

case3 = np.concatenate((asset1, asset2, asset3), axis=0)

# #case4
asset1 =        np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]
asset2 =        np.random.normal(1,   2,         int(1e+6))[np.newaxis,:]
asset3 = 0.01 * np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]

case4 = np.concatenate((asset1, asset2, asset3), axis=0)

# #case5
asset1 = np.random.normal(0,   1,         int(1e+6))[np.newaxis,:]
asset2 = np.random.normal(1,   2,         int(1e+6))[np.newaxis,:]
asset3 = np.random.normal(2,   1,         int(1e+6))[np.newaxis,:]

case5 = np.concatenate((asset1, asset2, asset3), axis=0)

cases = [case1, case2, case3, case4 ,case5]
Chart = np.zeros([5,5])

for i in range(len(cases)):
    results = Model.estimate(cases[i])
    Chart[i,:] = results

Chart_df = pd.DataFrame(Chart)
Chart_df.columns = ["g1(w)", "g2(w)", "g3(w)", "VaR", "CVaR"]
Chart_df.index = ["Case1", "Case2", "Case3", "Case4", "Case5"]

print(Chart_df)

