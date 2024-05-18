# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:28:47 2024

@author: Msantarin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

# Define a range for mu
mu = np.linspace(1.5, 1.9, 400)

# Calculate the likelihood
likelihood_out = likelihood_func(1.7, mu)

# Plotting
plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()
