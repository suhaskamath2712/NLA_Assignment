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
print("Calculating energy retained for different values of k to find minimum k for 90% energy retention in all channels.")
epsilon = 0.9
for k in range(0, 1961):
    energy_red = np.sum(S_red[:k]**2) / np.sum(S_red**2)
    energy_green = np.sum(S_green[:k]**2) / np.sum(S_green**2)
    energy_blue = np.sum(S_blue[:k]**2) / np.sum(S_blue**2)
    
    #output energy retained for each channel
    print(f"k={k}: Energy Retained - Red: {energy_red:.4f}, Green: {energy_green:.4f}, Blue: {energy_blue:.4f}")

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
