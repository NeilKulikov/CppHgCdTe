# CppHgCdTe

The tiny library designed to perform calculations of 
energy-momentum law in HgCdTe heterostructures.

*Main features:*
1. Uses Kane 8x8 hamiltonian in Burt-Foreman model with strain (in thin-film approx.)
2. Allows to compute 2D E(k) relations
3. Cross-platform library - there is no platform-specific dependencies
4. The result is standalone shared library & python bindings to call it

*Dependencies:*
1. GNU Scientific Library
2. Intel PSTL (optional, required for parallelism)
3. Numpy (optional, required for Python wrapper)
4. Ctypes (optional, required for Python wrapper)
