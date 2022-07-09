import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
import pdb
x = np.array([10, 14, 22, 24, 25, 26, 27, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64])
y = np.array([53.14267749, 93.1124486, 41.76301494, 59.72785568, 58.18433896, 58.12835186,
 54.33762812, 51.68721698, 51.6560659,  51.73072342, 50.9824215,  50.32227393,
 50.31967303, 51.08038614, 49.67096159, 49.67353615, 49.03186524, 49.03348584,
 49.81220867, 48.2625082, ])
plt.figure(figsize=(16, 9))


x = x.reshape(-1, 1)

start = 1
for eps in [1.01, 1.5, 1.75, 1.9, 3, 5]:
    plt.subplot(2, 3, start)
    start += 1
    huber = HuberRegressor(epsilon=eps).fit(x, y)

    w, b = huber.coef_, huber.intercept_
    mask = huber.outliers_

    reg = w * x + b
    plt.plot(x, reg, label=f'eps={eps}')
    for outlier, x_data, y_data in zip(mask, x, y):
        plt.scatter(x_data, y_data, c='k')
        if outlier:
            plt.scatter(x_data, y_data, c='r', marker='x', s=30)
    plt.legend()
plt.show()
