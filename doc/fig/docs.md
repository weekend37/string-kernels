# Documentation

## kernels.string_kernel

**Wrapper for a singly vectorized linear time string kernel implentation for data matrices X and Y**
```python
    Parameters
        - normalzie : bool, default=True
            indicates if the kernel output should be normalized s.t. max(K) <= 1
        - n_jobs : int, default=None
            how many CPUs to distribute the process over. If None, use maximum available CPUs.
    
    Returns
        - string_kernel_func : function
            function that takes in two data matrices X and Y as arguments
            (np.ndarray's of shapes (NX,MX) and (NY, MY) where N_ is the number of samples and M_ is sequence length)
            and returns the string kernel value between product of all samples in X and Y (int, float depending on normalization)
```

**Example**

```python
from sklearn import svm
from stringkernels.kernels import string_kernel
model = svm.SVC(kernel=string_kernel(n_jobs=32))
```

## kernels.polynomial_string_kernel

**Wrapper for a linear time polynomial string kernel distance implentation for two data matrices X and Y for a monomial with exponent p to run across n_jobs different CPUs.**
```python
    Parameters
        - p: float or int, default = 1.2
            exponent of the monomial which will be used
        - normalzie : bool, default=True
            indicates if the kernel output should be normalized s.t. max(K) <= 1
        - n_jobs : int, default=None
            how many CPUs to distribute the process over. If None, use maximum available CPUs.
    
    Returns
        - polynomial_string_kernel_func : function
            function that takes in two data matrices X and Y as arguments
            (np.ndarray's of shapes (NX,MX) and (NY, MY) where N_ is the number of samples and M_ is sequence length)
            and returns the polynomial string kernel value between product of all samples in X and Y (float)

```

**Example**

```python
from sklearn import svm
from stringkernels.kernels import polynomial_string_kernel
model = svm.SVC(kernel=polynomial_string_kernel(p=1.1))
```