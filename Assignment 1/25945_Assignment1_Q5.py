import numpy as np

#generate random matrix A
A = np.random.randn(100,2)

#this array keeps track of maximum 1-norm, 2-norm etc. of the vector Ax
#max_norm[0] represents maximum 1-norm of Ax, max_norm[1] represents maximum 2-norm of Ax, etc
max_norm = [-1,-1,-1,-1,-1,-1,-1]

for _ in range(10000):
    #Generate random x vector
    #The np.random.randn() method returns a matrix of 2x1 instead of a vector
    #the squeeze function is used to convert the 2x1 matrix to a vector
    x = np.squeeze(np.random.randn(2,1))                 

    #take 1-norm to 6-norm 
    #x is converted to unit vector for the particular norm
    #i.e if 1-norm of Ax is to be calculated, x is normalised by its 1-norm, and so on
    for p in range(1,7):
        #Calculate the relevant norm of x and normalise x accordingly
        pnorm_x = np.linalg.norm(x,ord=p)
        unit_x = [element/pnorm_x for element in x]

        #Calculate Ax
        #Again the @ operator produces a matrix, which needs to be squeezed to a vector 
        Ax = np.squeeze(A @ unit_x)
        
        #Update the max_norm 
        max_norm[p-1]=max(max_norm[p-1], np.linalg.norm(Ax, ord=p))
    
    #take infinity norm
    #This section is outside the loop because p cannot be suddenly assigned to infinity
    pnorm_x = np.linalg.norm(x,ord=np.inf)
    unit_x = [element/pnorm_x for element in x]
    Ax = np.squeeze(A @ unit_x)
    max_norm[6]=max(max_norm[6], np.linalg.norm(Ax, ord=np.inf))
    

print("1-norm of A =", np.linalg.norm(A, ord=1))
print("Max 1-norm of Ax =", max_norm[0], end="\n")

print("2-norm of A =", np.linalg.norm(A, ord=2))
print("Max 2-norm of Ax =", max_norm[1], end="\n")

print("infinity-norm of A =", np.linalg.norm(A, ord=np.inf))
print("Max infinity-norm of Ax =", max_norm[6])

print("Remaining norms of x: ", max_norm[2:6])