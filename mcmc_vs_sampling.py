import numpy as np
import matplotlib.pyplot as plt

def log_posterior(x, y):
    """
    Logarithm of the posterior distribution (up to a constant).
    """
    return -0.5 * ((y - x**3)**2 + x**2)

def rwm_sampler(y, num_samples, proposal_std, initial_x=0.0):

    samples = np.zeros(num_samples)
    x_current = initial_x
    accept_count = 0
    
    for i in range(num_samples):
        # Propose a new sample
        x_proposal = np.random.normal(x_current, proposal_std)
        
        # Calculate acceptance probability
        log_alpha = log_posterior(x_proposal, y) - log_posterior(x_current, y)
        alpha = np.exp(log_alpha)
        
        # Accept or reject
        if np.random.rand() < alpha:
            x_current = x_proposal
            accept_count += 1
        
        samples[i] = x_current
    
    acceptance_rate = accept_count / num_samples
    return samples, acceptance_rate

def importance_sampler(y, num_samples):
    """
    Importance sampler using the prior (Gaussian) as the proposal distribution.
    
    Parameters:
    - y: Observed value.
    - num_samples: Number of samples to draw.
    
    Returns:
    - samples: Samples drawn from the prior (proposal distribution).
    - weights: Importance weights for the samples.
    """
    # Proposal distribution: Gaussian with mean 0, variance 1
    samples = np.random.normal(0, 1, num_samples)
    
    # Target (unnormalized posterior): exp(-0.5 * [(y - x^3)^2 + x^2])
    log_weights = -0.5 * ((y - samples**3)**2 + samples**2)
    
    # Proposal (prior) density: exp(-0.5 * x^2)
    log_proposal = -0.5 * samples**2
    
    # Importance weights
    log_importance_weights = log_weights - log_proposal
    importance_weights = np.exp(log_importance_weights - np.max(log_importance_weights))  # Normalize for numerical stability
    
    return samples, importance_weights

# Parameters
y = -2  # Observed value
num_samples = 10000  # Number of samples for both methods
proposal_std = 1  # Proposal standard deviation for MCMC
initial_x = 0.0  # Initial value for x in MCMC

# Run MCMC sampler
samples_mcmc, acceptance_rate = rwm_sampler(y, num_samples, proposal_std, initial_x)
print(f"MCMC Acceptance rate: {acceptance_rate:.2f}")

# Run Importance sampler
importance_samples, importance_weights = importance_sampler(y, num_samples)

# Normalize weights for plotting
normalized_weights = importance_weights / np.sum(importance_weights)

# Plot histograms for MCMC and Importance Sampling
plt.figure(figsize=(12, 6))

# MCMC samples
plt.hist(samples_mcmc, bins=100, density=True, alpha=0.3, label="MCMC Samples")

# Weighted histogram for Importance Sampling
plt.hist(importance_samples, bins=100, weights=normalized_weights, density=True, alpha=0.4, label="Importance Samples")

# Add legend and labels
plt.title("MCMC vs Importance Sampling")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
