import numpy as np
import multiprocessing as mp
from functools import partial

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
