# String Kernels

This package contains an implementation of the **Polynomial String Kernel** and a linear time **String Kernel** algorithm as described in our paper, [High Resolution Ancestry Deconvolution for Next Generation Genomic Data](https://www.biorxiv.org/content/10.1101/2021.09.19.460980v1). 


<img caption="String Kernel Computations" src="https://raw.githubusercontent.com/weekend37/string-kernels/master/doc/fig/triangular_numbers.png">

It offers

- Linear time computation of two effective string kernels.

- Compatibility with Scikit-Learn's Support Vector Machines (easy plug-in).

- Multithreading.

## Usage

To install the package, execute from the command line

```
pip install string-kernels
```

And then you're all set!

Assuming you have [Scikit-Learn](https://scikit-learn.org/) already installed, you can use Lodhi's string kernel via

```python
from sklearn import svm
from stringkernels.kernels import string_kernel
model = svm.SVC(kernel=string_kernel)
```

and the polynomial string kernel,

```python
from sklearn import svm
from stringkernels.kernels import polynomial_string_kernel
model = svm.SVC(kernel=polynomial_string_kernel)
```

See the notebook [example.ipynb](https://github.com/weekend37/string-kernels/blob/master/example.ipynb) for further demonstration of usage.

If you end up using this in your research we kindly ask you to cite us! :)
