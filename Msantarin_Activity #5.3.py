import scipy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

# Define a range for mu
mu = np.linspace(1.65, 1.8, num=50)

# Define the likelihood function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.2)
    return likelihood_out

# Calculate the likelihood for each mu
likelihood_out = likelihood_func(1.7, mu)

# Define the uniform distribution
uniform_dist = sts.uniform.pdf(mu, loc=1.65, scale=0.2) + 1  # Adjusted to span the range of mu
uniform_dist = uniform_dist / uniform_dist.sum()  # Normalize the uniform distribution

# Calculate the unnormalized posterior
unnormalized_posterior = likelihood_out * uniform_dist

# Plotting the unnormalized posterior
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.title("Unnormalized Posterior Distribution")
plt.show()
