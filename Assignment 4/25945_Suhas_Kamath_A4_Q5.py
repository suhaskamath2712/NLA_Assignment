import numpy as np
import matplotlib.pyplot as plt
import os

def f(t):
    """The function to be approximated."""
    return np.sin(10 * t)

def back_substitution(R, y):
    """
    Solves the upper triangular system Rx = y using back substitution.
    """
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x

def modified_gram_schmidt(A):
    """Performs QR factorization using the Modified Gram-Schmidt process."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()

    for k in range(n):
        R[k, k] = np.linalg.norm(V[:, k])
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k + 1, n):
            R[k, j] = np.dot(Q[:, k].T, V[:, j])
            V[:, j] = V[:, j] - R[k, j] * Q[:, k]
    return Q, R

def householder_factorization(A):
    """Performs QR factorization using Householder reflections."""
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.copysign(np.linalg.norm(x), x[0])
        v = (x + e1)
        v = v / np.linalg.norm(v)

        R[k:, :] -= 2 * np.outer(v, v.T @ R[k:, :])
        Q[:, k:] -= 2 * Q[:, k:] @ np.outer(v, v.T)

    return Q[:,:n], R[:n,:]

# --- 1. Generate Data and Vandermonde Matrix ---
m = 100
degree = 14
domain = np.linspace(0, 1, m)
data_points = f(domain)
A = np.vander(domain, degree + 1)

# --- 2. Solve the Least Squares Problem using all 5 methods ---

# Inbuilt Python "True" Solution (Baseline for comparison)
true_coeffs = np.linalg.lstsq(A, data_points, rcond=None)[0]
print("Coefficients using np.linalg.lstsq ('True' Solution) calculated.")

# Method (a): Modified Gram-Schmidt
Q_mgs, R_mgs = modified_gram_schmidt(A)
mgs_coeffs = back_substitution(R_mgs, Q_mgs.T @ data_points)
print("Coefficients using Modified Gram-Schmidt calculated.")

# Method (b): Householder Factorization
Q_hh, R_hh = householder_factorization(A)
householder_coeffs = back_substitution(R_hh, Q_hh.T @ data_points)
print("Coefficients using Householder Factorization calculated.")

# Method (c): SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
svd_coeffs = VT.T @ ((U.T @ data_points) / S)
print("Coefficients using SVD calculated.")

# Method (d): Normal Equations
A_T_A = A.T @ A
A_T_b = A.T @ data_points
normal_eq_coeffs = np.linalg.solve(A_T_A, A_T_b)
print("Coefficients using Normal Equations calculated.")

# --- 3. Plot and Save Individual Comparisons ---

# Directory to save plots. Change this if you get a PermissionError.
save_dir = r"H:\My Drive\Numerical Linear Algebra\Assignments\Assignment 4"
print(f"\nAttempting to save plots to: {save_dir}")

# Plot 1: Modified Gram-Schmidt vs. True Solution
plt.figure(figsize=(19.2, 10.8))  # 4K resolution size
plt.title('Modified Gram-Schmidt vs. True Solution', fontsize=14)
plt.plot(domain, data_points, 'ko', markersize=5, label='Original Data')
plt.plot(domain, np.polyval(true_coeffs, domain), 'b-', label='"True" Solution (lstsq)')
plt.plot(domain, np.polyval(mgs_coeffs, domain), 'g--', label='Modified Gram-Schmidt')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'mgs_comparison.png'), dpi=400) # Added dpi=400 for 4K resolution
#plt.show()

# Plot 2: Householder vs. True Solution
plt.figure(figsize=(19.2, 10.8))  # 4K resolution size
plt.title('Householder Factorization vs. True Solution', fontsize=14)
plt.plot(domain, data_points, 'ko', markersize=5, label='Original Data')
plt.plot(domain, np.polyval(true_coeffs, domain), 'b-', label='"True" Solution (lstsq)')
plt.plot(domain, np.polyval(householder_coeffs, domain), 'g--.', label='Householder')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'householder_comparison.png'), dpi=400) # Added dpi=400
#plt.show()

# Plot 3: SVD vs. True Solution
plt.figure(figsize=(19.2, 10.8))  # 4K resolution size
plt.title('SVD vs. True Solution', fontsize=14)
plt.plot(domain, data_points, 'ko', markersize=5, label='Original Data')
plt.plot(domain, np.polyval(true_coeffs, domain), 'b-', label='"True" Solution (lstsq)')
plt.plot(domain, np.polyval(svd_coeffs, domain), 'g--', label='SVD')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'svd_comparison.png'), dpi=400) # Added dpi=400
#plt.show()

# Plot 4: Normal Equations vs. True Solution
plt.figure(figsize=(19.2, 10.8))  # 4K resolution size
plt.title('Normal Equations vs. True Solution', fontsize=14)
plt.plot(domain, data_points, 'ko', markersize=5, label='Original Data')
plt.plot(domain, np.polyval(true_coeffs, domain), 'b-', label='"True" Solution (lstsq)')
plt.plot(domain, np.polyval(normal_eq_coeffs, domain), 'm-', label='Normal Equations')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'normal_equations_comparison.png'), dpi=400) # Added dpi=400
#plt.show()

print(f"\nSuccessfully saved 4 high-resolution plots to the '{save_dir}' directory.")

#Get least square errors for all methods
true_error = np.linalg.norm(data_points - np.polyval(true_coeffs, domain))
mgs_error = np.linalg.norm(data_points - np.polyval(mgs_coeffs, domain))
householder_error = np.linalg.norm(data_points - np.polyval(householder_coeffs, domain))
svd_error = np.linalg.norm(data_points - np.polyval(svd_coeffs, domain))
normal_eq_error = np.linalg.norm(data_points - np.polyval(normal_eq_coeffs, domain))

print("\nLeast Squares Errors:")
print(f"True Solution (lstsq): {true_error:.6e}")
print(f"Modified Gram-Schmidt: {mgs_error:.6e}")
print(f"Householder: {householder_error:.6e}")
print(f"SVD: {svd_error:.6e}")
print(f"Normal Equations: {normal_eq_error:.6e}")