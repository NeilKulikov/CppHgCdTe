# CppHgCdTe

The tiny library designed to perform calculations of 
energy-momentum law in HgCdTe heterostructures.

*Main features:*
1. Uses Kane 8x8 hamiltonian in Burt-Foreman model with strain (in thin-film approx.)
2. Allows to compute 2D E(k) relations
3. Cross-platform library - there is no platform-specific dependencies
4. The results are standalone shared library & python bindings to call it

*Dependencies:*
1. GNU Scientific Library
2. Intel PSTL (optional, required for parallelism)
3. Numpy (optional, required for Python wrapper)
4. Ctypes (optional, required for Python wrapper)

*Building:*
1. Make sure that the GSL library, cmake, make & any compiler containing a standard C++17 are already installed
2. Create directory for building: ```mkdir build && cd build```
3. Run cmake to configure building: ```cmake ..```
4. Build package, using make: ``` make```
5. *Optional:* run tests: ```ctest --extra-verbose```
6. Now you can find compiled library in ```build/bin/lib/libinterop.*dylib/*so/*dll```

If you have any questions - please contact me via [e-mail](mailto:neilkulikov@gmail.com).
