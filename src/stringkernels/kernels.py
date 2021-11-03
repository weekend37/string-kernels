import numpy as np
import multiprocessing as mp
from functools import partial

## ------------------------------- String Kernel ------------------------------- ##

def string_kernel_vectors(x,y,normalize=True):
    """
    Linear time string kernel distance implentation for two sequences x and y
    """
    z = x==y # O(M)
    tri, K = 0, 0
    for equal in z: # O(M)
        if equal: # (1)
            tri += 1 # O(1)
            K += tri # O(1)
        else:
            tri = 0 # O(1)

    if normalize:
        K = K / len(z)*(len(z)-1)//2
    return K

def string_kernel_vectorized(x,Y,normalize=True):
    """
    Vectorized linear time string kernel implentation for a sequence x and data matrix Y
    """
    z = x==Y
    tri = np.zeros(len(z), dtype=int)
    K = np.zeros(len(z), dtype=int)
    for equal in z.T:
        tri += equal
        tri[~equal] = 0 
        K += tri
    
    if normalize:
        K = K / len(z)*(len(z)-1)//2

    return K

def string_kernel_singlethread(X,Y,normalize=True):
    """
    Singly vectorized linear time string kernel implentation for data matrices X and Y
    """
    return np.array([string_kernel_vectorized(x,Y,normalize=normalize) for x in X])

def string_kernel(X,Y,normalize=True,n_jobs=None):
    """
    Singly vectorized linear time string kernel implentation for data matrices X and Y with multithreading
    """
    with mp.Pool(n_jobs) as pool:
        multithread_func = partial(string_kernel_vectorized,Y=Y,normalize=normalize)
        K_list = pool.map(multithread_func, X)
    
    K = np.array(K_list).squeeze()
    
    return K

## ------------------------------- Polynomial String Kernel ------------------------------- ##

def polynomial_string_kernel_vectors(x,y,p,normalize=False):
    """
    Linear time polynomial string kernel distance implentation for two sequences x and y
    for a monomial with exponent p
    """
    z = x==y # O(M)
    contigs = []
    counter = 0
    for equal in z: # O(M)
        if equal: # (1)
            counter += 1 # O(1)
        else:
            contigs += [counter] # O(1)
            counter = 0 # O(1)
    contigs += [counter] # O(1)
    contigs = np.array(contigs) # O(1)
    K = np.sum(contigs**p) # O(2M)

    if normalize:
        assert p >= 1, "Can't normalize for monomials with exponent less than 1"
        K = K / len(x)**p

    return K

def polynomial_string_kernel_singlethread(X,Y,p=1.2,normalize=False):
    """
    Linear time polynomial string kernel distance implentation for two data matrices X and Y
    for a monomial with exponent p
    """
    K = np.zeros((len(X), len(Y)), dtype=int)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i,j] = polynomial_string_kernel_vectors(x,y,p=p,normalize=normalize)
            
    return K

def polynomial_string_kernel(X,Y,p=1.2,n_jobs=16,normalize=False):
    """
    Multithreaded linear time polynomial string kernel distance implentation for two data matrices X and Y
    for a monomial with exponent p to run across n_jobs different cpus.
    """
    Xn, Xm = X.shape
    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(partial(polynomial_string_kernel_singlethread,Y=Y,p=p,normalize=normalize), X.reshape(Xn, 1, Xm))
    
    K = np.array(K_list).squeeze()
    
    return K
