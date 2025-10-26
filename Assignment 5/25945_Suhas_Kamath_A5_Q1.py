import numpy as np
import matplotlib.pyplot as plt
import os

# --- Build the Markov transition matrix M ---
M = np.zeros((10, 10))

# Outgoing links dictionary: for each node j, list nodes i that j links to.
# Interpretation: j -> i for each i in links[j].
links = {
    0: [1, 2], 1: [2, 5], 2: [0, 1], 3: [4], 4: [5, 7],
    5: [6, 9], 6: [2, 7], 7: [4, 5], 8: [1, 7], 9: [8]
}

# Fill M so each nonzero column j distributes mass uniformly over the
# outgoing links of j. For example, if node j points to k_j targets,
# each target i receives 1/k_j probability from column j.
for j, i_list in links.items():
    k_j = len(i_list)
    for i in i_list:
        M[i, j] = 1.0 / k_j

# --- Power iteration parameters and initialization ---
max_iterations = 100       # hard cap to avoid infinite loops
tolerance = 1e-6           # stopping criterion on eigenvector change

# Random initial vector (nonzero), normalized to unit 2-norm.
x_k = np.random.rand(M.shape[0])
x_k /= np.linalg.norm(x_k)

# Containers to store convergence diagnostics for plotting/analysis
residuals = []                 # ||M x_k - lambda_k x_k||_2
eigenvector_differences = []   # ||x_{k+1} - x_k||_2
eigenvalues = []               # Rayleigh quotient estimates

print("Starting power iteration...")

# --- Power iteration loop ---
for k in range(max_iterations):
    # Apply the matrix and renormalize to keep vector magnitude bounded
    x_k_plus_1 = M @ x_k
    x_k_plus_1 /= np.linalg.norm(x_k_plus_1)

    # Rayleigh quotient provides an estimate of the eigenvalue for x_{k+1}
    eigenval = x_k_plus_1.T @ M @ x_k_plus_1
    eigenvalues.append(eigenval)

    # Residual norm: measures how well (lambda, x) satisfies the eigen eqn
    residuals.append(np.log(np.linalg.norm(M @ x_k_plus_1 - eigenval * x_k_plus_1)))

    # Change in eigenvector between iterations (useful stopping metric)
    eigenvector_differences.append(np.linalg.norm(x_k_plus_1 - x_k))

    x_k = x_k_plus_1

    # Check convergence
    if eigenvector_differences[-1] < tolerance:
        print(f"\nConverged after {k+1} iterations.")
        break

# PageRank interpretation: normalize so components sum to 1 (probability)
pagerank_vector = x_k / np.linalg.norm(x_k)

print(pagerank_vector)

print(f"Highest PageRank: Node {np.argmax(pagerank_vector)} (Rank: {np.max(pagerank_vector):.4f})")
print(f"Lowest PageRank: Node {np.argmin(pagerank_vector)} (Rank: {np.min(pagerank_vector):.4f})")

save_dir = r"H:\My Drive\Numerical Linear Algebra\Assignments\Assignment 5"
print(f"\nAttempting to save plots to: {save_dir}")

#Plot 1
plt.figure(figsize=(16,9))
plt.plot(residuals)
plt.title('Convergence of Residual Norm in Power Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Residual Norm ||M x_k - λ_k x_k||_2')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'residual_norm.png'), dpi=400)


#Plot 2
plt.figure(figsize=(16,9))
plt.plot(eigenvector_differences)
plt.title('Convergence of Eigenvector Differences in Power Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Eigenvector Difference ||x_{k+1} - x_k||_2')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'eigenvector_differences.png'), dpi=400)


#Plot 3
plt.figure(figsize=(16,9))
plt.plot(eigenvalues)
plt.title('Convergence of Rayleigh Coefficients in Power Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Rayleigh Quotient (Estimate of Eigenvalue)')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'rayleigh_coefficients.png'), dpi=400)

print("\nAll plots saved.")