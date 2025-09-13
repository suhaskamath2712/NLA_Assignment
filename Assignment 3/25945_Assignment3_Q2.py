import cv2
import numpy as np

# Load the image
image = np.asarray(cv2.imread(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\SMACS_0723.png"))

image_red = image[:,:,0]
image_green = image[:,:,1]
image_blue = image[:,:,2]

#print array dimensions
print("Red Channel Dimensions: ", image_red.shape)
print("Green Channel Dimensions: ", image_green.shape)
print("Blue Channel Dimensions: ", image_blue.shape)

#Perform SVD on each channel
U_red, S_red, VT_red = np.linalg.svd(image_red , full_matrices=False)
U_green, S_green, VT_green = np.linalg.svd(image_green , full_matrices=False)
U_blue, S_blue, VT_blue = np.linalg.svd(image_blue , full_matrices=False)

#Save SVD Components to CSV
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\U_red_matrix.csv", U_red, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\S_red_matrix.csv", S_red, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\VT_red_matrix.csv", VT_red, delimiter=",")

np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\U_green_matrix.csv", U_green, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\S_green_matrix.csv", S_green, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\VT_green_matrix.csv", VT_green, delimiter=",")

np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\U_blue_matrix.csv", U_blue, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\S_blue_matrix.csv", S_blue, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\VT_blue_matrix.csv", VT_blue, delimiter=",")

'''
#Perform SVD
U, S, VT = np.linalg.svd(image , full_matrices=False)

print("Original Matrix: ")
print(np.asarray(image))

print ("Left Singular Vectors (U): ")
print(U)
print("Singular Values (S): ")
print(S)
print("Right Singular Vectors (Vt): ")
print(VT)

#Save SVD Components to CSV
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\U_matrix.csv", U, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\S_matrix.csv", S, delimiter=",")
np.savetxt(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\VT_matrix.csv", VT, delimiter=",")


#Low rank approximation
k = 50  # Number of singular values to keep
U_k = U[:, :k]
S_k = S[:, :k]
VT_k = VT[:k, :]

# Reconstruct the image using the top k singular values
S_k_matrix = np.diag(S_k)
image_approx = np.dot(U_k, np.dot(S_k_matrix, VT_k))
image_approx = np.clip(image_approx, 0, 255).astype(np.uint8)
print("Low Rank Approximation: ")
cv2.imwrite(r"H:\\My Drive\\Numerical Linear Algebra\\Assignments\\Assignment 3\\SMACS_0723_low_rank_approx.png", image_approx)
'''