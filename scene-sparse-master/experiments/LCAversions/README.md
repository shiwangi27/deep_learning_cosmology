LCAversions
===========

Locally Competitive Algorithm written in Python using various packages.

Current implementations include:

* LCAnumpy: Pure python+numpy. Requires numpy.
* LCAfortan (float64 only): Fortran 90 with python wrapper. Requires f2py.
* LCAnumbaprog(float32 only): NumbaPro GPU implementation. Requires numbapro.

The Fortran based version can be compiled using the command:
```
f2py -c -m lca lca.f90
```
in the LCAfortran folder.

To run tests, do:
```
nosetests
```
from the base directory.

### Speedup on Tesla K40 vs. Xeon E-5 1620 (with MKL)
```
Parameters:
             numDict: 4096
             numBatch: 128
             dataSize: 256
             nIter: 300

---------------Numpy based LCA----------------
Average time over 1 trials:
30.818508 s
---------------Fortran based LCA--------------
Average time over 1 trials:
16.504698 s
----------------GPU based LCA-----------------
Average time over 6 trials:
0.987908 s
```
