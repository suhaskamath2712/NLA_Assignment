import cv2
import numpy as np

# Load the image
image = np.asarray(cv2.imread(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\SMACS_0723.png"))

image_red = image[:,:,0]
image_green = image[:,:,1]
image_blue = image[:,:,2]

#Perform SVD on each channel
print("Performing SVD on each component: ")
U_red, S_red, VT_red = np.linalg.svd(image_red , full_matrices=False)
U_green, S_green, VT_green = np.linalg.svd(image_green , full_matrices=False)
U_blue, S_blue, VT_blue = np.linalg.svd(image_blue , full_matrices=False)

print("SVD Completed.")
print("Starting low rank approximation")

#Calculate energy retained and hence appropriate k
#Energy retained = sum of squares of first k singular values / sum of squares of all singular values
#We need to find the minimum k such that energy retained >= 0.9 for all channels
epsilon = 0.9

print(f"Calculating energy retained for different values of k to find minimum k for {epsilon*100}% energy retention in all channels.")

#Calculate total energy for each channel
total_energy_red = np.sum(S_red**2)
total_energy_green = np.sum(S_green**2)
total_energy_blue = np.sum(S_blue**2)

for k in range(0, 1961):
    energy_red = np.sum(S_red[:k]**2) / total_energy_red
    energy_green = np.sum(S_green[:k]**2) / total_energy_green
    energy_blue = np.sum(S_blue[:k]**2) / total_energy_blue

    if energy_red >= epsilon and energy_green >= epsilon and energy_blue >= epsilon:
        print(f"Minimum k to retain at least 90% energy in all channels: {k}")
        break

#Low rank approximation
U_red_k = U_red[:, :k]
S_red_k = np.diag(S_red[:k])
VT_red_k = VT_red[:k, :]

U_green_k = U_green[:, :k]
S_green_k = np.diag(S_green[:k])
VT_green_k = VT_green[:k, :]

U_blue_k = U_blue[:, :k]
S_blue_k = np.diag(S_blue[:k])
VT_blue_k = VT_blue[:k, :]

print("Low rank approximation completed.")
print("Reconstructing the image from low rank approximation")
#Reconstruct the image with reduced rank
image_red_k = np.dot(U_red_k, np.dot(S_red_k, VT_red_k))
image_green_k = np.dot(U_green_k, np.dot(S_green_k, VT_green_k))
image_blue_k = np.dot(U_blue_k, np.dot(S_blue_k, VT_blue_k))
image_reconstructed = np.zeros(image.shape)
image_reconstructed[:,:,0] = image_red_k
image_reconstructed[:,:,1] = image_green_k
image_reconstructed[:,:,2] = image_blue_k

image_reconstructed = np.clip(image_reconstructed, 0, 255).astype(np.uint8)
print("Image reconstruction completed.")

#Save the reconstructed image
cv2.imwrite(rf"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\SMACS_0723_reconstructed_k{k}.png", image_reconstructed)
print("Reconstructed image saved.")

#Error in image reconstruction for each channel using 2-norm
error_red_2 = np.linalg.norm(image_red - image_red_k, 2)
error_green_2 = np.linalg.norm(image_green - image_green_k, 2)
error_blue_2 = np.linalg.norm(image_blue - image_blue_k, 2)
print(f"Reconstruction Error (2-Norm) - Red: {error_red_2:.4f}, Green: {error_green_2:.4f}, Blue: {error_blue_2:.4f}")

#Error in image reconstruction for each channel using frobenius norm
error_red_fro = np.linalg.norm(image_red - image_red_k, 'fro')
error_green_fro = np.linalg.norm(image_green - image_green_k, 'fro')
error_blue_fro = np.linalg.norm(image_blue - image_blue_k, 'fro')
print(f"Reconstruction Error (Frobenius Norm) - Red: {error_red_fro:.4f}, Green: {error_green_fro:.4f}, Blue: {error_blue_fro:.4f}")

#Print (k+1)th singular value for each channel
print(f"(k+1)th Singular Value - Red: {S_red[k]:.4f}")
print(f"(k+1)th Singular Value - Green: {S_green[k]:.4f}")
print(f"(k+1)th Singular Value - Blue: {S_blue[k]:.4f}")

red_diff = 0
green_diff = 0
blue_diff = 0

#Sum of squares of singular values from (k+1) to end for each channel
for v in range(k, len(S_red)):
    red_diff += S_red[v] ** 2
    green_diff += S_green[v] ** 2
    blue_diff += S_blue[v] ** 2

print(f"Sq. root of sum of squares of singular values from (k+1) to end - Red: {np.sqrt(red_diff):.4f}")
print(f"Sq. root of sum of squares of singular values from (k+1) to end - Green: {np.sqrt(green_diff):.4f}")
print(f"Sq. root of sum of squares of singular values from (k+1) to end - Blue: {np.sqrt(blue_diff):.4f}")

