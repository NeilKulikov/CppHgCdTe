# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/neilkulikov/Documents/SWork/task001b

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/neilkulikov/Documents/SWork/task001b/build

# Include any dependencies generated for this target.
include build/test/CMakeFiles/test2.dir/depend.make

# Include the progress variables for this target.
include build/test/CMakeFiles/test2.dir/progress.make

# Include the compile flags for this target's objects.
include build/test/CMakeFiles/test2.dir/flags.make

build/test/CMakeFiles/test2.dir/splines.cpp.o: build/test/CMakeFiles/test2.dir/flags.make
build/test/CMakeFiles/test2.dir/splines.cpp.o: ../test/splines.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/neilkulikov/Documents/SWork/task001b/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object build/test/CMakeFiles/test2.dir/splines.cpp.o"
	cd /Users/neilkulikov/Documents/SWork/task001b/build/build/test && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test2.dir/splines.cpp.o -c /Users/neilkulikov/Documents/SWork/task001b/test/splines.cpp

build/test/CMakeFiles/test2.dir/splines.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test2.dir/splines.cpp.i"
	cd /Users/neilkulikov/Documents/SWork/task001b/build/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/neilkulikov/Documents/SWork/task001b/test/splines.cpp > CMakeFiles/test2.dir/splines.cpp.i

build/test/CMakeFiles/test2.dir/splines.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test2.dir/splines.cpp.s"
	cd /Users/neilkulikov/Documents/SWork/task001b/build/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/neilkulikov/Documents/SWork/task001b/test/splines.cpp -o CMakeFiles/test2.dir/splines.cpp.s

# Object files for target test2
test2_OBJECTS = \
"CMakeFiles/test2.dir/splines.cpp.o"

# External object files for target test2
test2_EXTERNAL_OBJECTS =

bin/test2: build/test/CMakeFiles/test2.dir/splines.cpp.o
bin/test2: build/test/CMakeFiles/test2.dir/build.make
bin/test2: /usr/local/lib/libboost_unit_test_framework.dylib
bin/test2: /usr/local/Cellar/gsl/2.6/lib/libgsl.dylib
bin/test2: /usr/local/Cellar/gsl/2.6/lib/libgslcblas.dylib
bin/test2: build/test/CMakeFiles/test2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/neilkulikov/Documents/SWork/task001b/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/test2"
	cd /Users/neilkulikov/Documents/SWork/task001b/build/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
build/test/CMakeFiles/test2.dir/build: bin/test2

.PHONY : build/test/CMakeFiles/test2.dir/build

build/test/CMakeFiles/test2.dir/clean:
	cd /Users/neilkulikov/Documents/SWork/task001b/build/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test2.dir/cmake_clean.cmake
.PHONY : build/test/CMakeFiles/test2.dir/clean

build/test/CMakeFiles/test2.dir/depend:
	cd /Users/neilkulikov/Documents/SWork/task001b/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/neilkulikov/Documents/SWork/task001b /Users/neilkulikov/Documents/SWork/task001b/test /Users/neilkulikov/Documents/SWork/task001b/build /Users/neilkulikov/Documents/SWork/task001b/build/build/test /Users/neilkulikov/Documents/SWork/task001b/build/build/test/CMakeFiles/test2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : build/test/CMakeFiles/test2.dir/depend
