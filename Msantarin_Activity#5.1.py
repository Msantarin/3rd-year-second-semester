# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:50:49 2024

@author: Msantarin
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Define a range for mu
mu = np.linspace(1.65, 1.8, num=50)

# Define the range for uniform distribution
uniform_dist = sts.uniform.pdf(mu, loc=1.65, scale=0.2) + 1  # Adjusted to span the range of mu

# Normalize the uniform distribution to make the probability densities sum to 1
uniform_dist = uniform_dist / uniform_dist.sum()

# Define the beta distribution
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

# Plotting
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()
plt.show()
