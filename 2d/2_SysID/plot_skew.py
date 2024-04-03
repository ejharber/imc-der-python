
from scipy.stats import wasserstein_distance
from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt

# skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)

x = np.linspace(-10, 10, 1000)
out = skewnorm.pdf(x, a=10, loc=1, scale=1)
r = skewnorm.rvs(a=10, loc=1, scale=1, size=1000)
print(out)

plt.hist(r, bins=40, density=True)
plt.plot(x, out)
plt.show()

# print(wasserstein_distance([0, 1, 2], [2, 1, 0]))
