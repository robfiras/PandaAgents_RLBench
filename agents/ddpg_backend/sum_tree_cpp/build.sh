if [ -d "./externals" ]
then 
    echo "externals directy already exists. Start checking for pybind ..."
else
    mkdir externals
fi
if [ -d "./externals/pybind11" ]
then 
    echo "Pybind11 already downloaded."
else 
    cd externals
    wget --no-check-certificate  https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz
    mkdir pybind11
    tar xvzf v2.5.0.tar.gz -C pybind11 --strip-components 1
    cd ..
fi
if [ -d "./build" ]
then 
    echo "build directory already exists. Start building ..."
else
    mkdir build
    echo "Start building ..."
fi
cd build 
cmake ..
make 
cp sum_tree_cpp.*.so ..

