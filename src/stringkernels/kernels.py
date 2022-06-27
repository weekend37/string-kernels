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

def string_kernel_multithread(X,Y,normalize=True,n_jobs=None):
    """
    Singly vectorized linear time string kernel implentation for data matrices X and Y with multithreading
    """
    with mp.Pool(n_jobs) as pool:
        multithread_func = partial(string_kernel_vectorized,Y=Y,normalize=normalize)
        K_list = pool.map(multithread_func, X)
    
    K = np.array(K_list).squeeze()
    
    return K

def string_kernel(normalize=True,n_jobs=None):
    """
    Wrapper for a singly vectorized linear time string kernel implentation for data matrices X and Y
    -----------
    Parameters
        - normalzie : bool, default=True
            indicates if the kernel output should be normalized s.t. max(K) <= 1
        - n_jobs : int, default=None
            how many CPUs to distribute the process over. If None, use maximum available CPUs.
    -----------
    Returns
        - string_kernel_func : function
            function that takes in two data matrices X and Y as arguments
            (np.ndarray's of shapes (NX,MX) and (NY, MY) where N_ is the number of samples and M_ is sequence length)
            and returns the string kernel value between product of all samples in X and Y (int, float depending on normalization)
    """
    if n_jobs is not None and n_jobs==1:
        return partial(string_kernel_singlethread, normalize=normalize)
    else:
        return partial(string_kernel_multithread, normalize=normalize, n_jobs=n_jobs)

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

def polynomial_string_kernel_multithread(X,Y,p=1.2,normalize=False,n_jobs=16):
    """
    Multithreaded linear time polynomial string kernel distance implentation for two data matrices X and Y
    for a monomial with exponent p to run across n_jobs different cpus.
    """
    Xn, Xm = X.shape
    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(partial(polynomial_string_kernel_singlethread,Y=Y,p=p,normalize=normalize), X.reshape(Xn, 1, Xm))
    
    K = np.array(K_list).squeeze()
    
    return K

def polynomial_string_kernel(p=1.2,normalize=False, n_jobs=16):
    """
    Wrapper for a linear time polynomial string kernel distance implentation for two data matrices X and Y
    for a monomial with exponent p to run across n_jobs different cpus.
    -----------
    Parameters
        - p: float or int, default = 1.2
            exponent of the monomial which will be used
        - normalzie : bool, default=True
            indicates if the kernel output should be normalized s.t. max(K) <= 1
        - n_jobs : int, default=None
            how many CPUs to distribute the process over. If None, use maximum available CPUs.
    -----------
    Returns
        - polynomial_string_kernel_func : function
            function that takes in two data matrices X and Y as arguments
            (np.ndarray's of shapes (NX,MX) and (NY, MY) where N_ is the number of samples and M_ is sequence length)
            and returns the polynomial string kernel value between product of all samples in X and Y (float)
    """
    if n_jobs is not None and n_jobs==1:
        return partial(polynomial_string_kernel_singlethread, p=p, normalize=normalize)
    else:
        return partial(polynomial_string_kernel_multithread, p=p, normalize=normalize, n_jobs=n_jobs)

## ------------------------------- CovRSK ------------------------------- ##

def ohe(idx, size):
    out = np.zeros(size,dtype=int)
    out[idx-1] = 1
    return out

def CovSample(M, alpha, beta, seed=1):

    np.random.seed(seed)
    
    Ms = [1]
    for m in range(2,M+1): 
        if (1-(alpha**(m-Ms[-1]+1)))*(m**(-beta)) >= np.random.rand():
            Ms += [m]

    return Ms

def CovRSK_vectorized(x,Y,Ms_ohe):
    z = x==Y
    K, tri, cov_tri = np.zeros((3, len(z)), dtype=int)
    for equal in z.T:
        tri += equal
        mask = Ms_ohe[tri] == 1
        cov_tri += mask*equal
        tri[~equal] = 0
        cov_tri[~equal] = 0
        K += cov_tri
    return K
    
def CovRSK_singlethread(X, Y, alpha=0.6, beta=1.0, seed=37):

    Xn, Xm = X.shape
    Ms = CovSample(Xm, alpha, beta, seed)
    Ms_ohe = ohe(idx=np.array(Ms)+1, size=Xm+1)
    K = np.array([CovRSK_vectorized(x,Y,Ms_ohe=Ms_ohe) for x in X])

    return K

def CovRSK_multithread(X, Y, alpha=0.6, beta=1.0, n_jobs=None, seed=37):
    
    Xn, Xm = X.shape
    Ms = CovSample(Xm, alpha, beta, seed)
    Ms_ohe = ohe(idx=np.array(Ms)+1, size=Xm+1)
    func = partial(CovRSK_vectorized, Y=Y,Ms_ohe=Ms_ohe)
    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(func, X)
    
    K = np.array(K_list).squeeze()
    
    return K
    