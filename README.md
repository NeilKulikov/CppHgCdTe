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

*Building:*
0. Make sure that the GSL libraru is already installed
1. Create directory for building: ```mkdir build && cd build```
2. Run cmake to configure building: ```cmake ..```
3. Build package, using make: ``` make .```
4. *Optional:* run tests: ```ctest --extra-verbose```
5. Now you can find compiled library in ```build/bin/lib/libinterop.*dylib/*so/*dll```

If you have any questions - please contact me via [e-mail](mailto:neilkulikov@gmail.com).
