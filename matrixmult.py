import numpy as np
import random
import time

def generate_matrices(n):
    """
    Generates two random n x n matrices with values uniformly sampled from [-2, 2].

    Parameters:
    n (int): The size of the square matrices to generate.

    Returns:
    tuple: A tuple (A, B) where A and B are numpy arrays of shape (n, n).
    """
    np.random.seed(10)  # Set seed in order to ensure reproducibility
    A = np.random.uniform(-2, 2, (n, n))
    B = np.random.uniform(-2, 2, (n, n))
    return A, B

def naive_method(A, B):
    """
    Performs matrix multiplication using the naive triple-loop method.
    Also counts floating-point operations (FLOPs), assuming each multiply-add is 2 FLOPs.

    Parameters:
    A (ndarray): Left matrix of shape (n, n).
    B (ndarray): Right matrix of shape (n, n).

    Returns:
    tuple: A tuple (C, flops) where C is the result matrix and flops is the count of floating-point operations.
    """
    n = A.shape[0]
    C = np.zeros((n, n))
    
    # Count floating-point operations (FLOPs)
    flops = 0
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                flops += 2 
    
    return C, flops

def strassen_method(A, B):
    """
    Performs matrix multiplication using Strassen's recursive algorithm.
    Reduces the multiplication complexity to approximately O(n^2.81).
    Also counts floating-point operations (FLOPs).

    Parameters:
    A (ndarray): Left matrix of shape (n, n), where n is a power of 2.
    B (ndarray): Right matrix of shape (n, n), where n is a power of 2.

    Returns:
    tuple: A tuple (C, flops) where C is the result matrix and flops is the count of floating-point operations.
    """
    n = A.shape[0]
    # Base case for 2x2
    if n <= 2:
        return naive_method(A, B)
    # Divide matrices into quadrants
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:] 
    # Compute 7 matrix products
    flops = 0  
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12
    # Recursive multiplications
    P1, flops1 = strassen_method(A11, S1)
    P2, flops2 = strassen_method(S2, B22)
    P3, flops3 = strassen_method(S3, B11)
    P4, flops4 = strassen_method(A22, S4)
    P5, flops5 = strassen_method(S5, S6)
    P6, flops6 = strassen_method(S7, S8)
    P7, flops7 = strassen_method(S9, S10)
    # Update FLOPs count
    flops = flops1 + flops2 + flops3 + flops4 + flops5 + flops6 + flops7 + 10 * (n // 2) ** 2
    # Compute resuls to form C
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7
    # Combine quadrants to form C
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    return C, flops

n = 2**10  # 1024
    
# Generate matrices
A, B = generate_matrices(n)
    
# Naive method
print("Naive Method Matrix Multiplication:")
start_time = time.time()
C_naive, naive_flops = naive_method(A, B)
naive_time = time.time() - start_time
print(f"Time: {naive_time:.4f} seconds")
print(f"Floating-point operations: {naive_flops}")
# Strassen method
print("Strassen Method Matrix Multiplication:")
start_time = time.time()
C_strassen, strassen_flops = strassen_method(A, B)
strassen_time = time.time() - start_time
print(f"Time: {strassen_time:.4f} seconds")
print(f"Floating-point operations: {strassen_flops}")
