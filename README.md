# HOW TO BUILD AND INSTALL
Assuming functional pytorch installation.
cd into pyg-rocm-build, then ```bash ./build_and_install_pyg.sh```
THIS WILL BUILD WHEELS *AND* INSTALL THEM.
NOTE: This will only build libraries that have not yet been installed. To rebuild existing libraries, use the --force-rebuild option.

To rebuild all libraries, use:
```bash ./build_and_install_pyg.sh --force-rebuild=all```

To rebuild specific libraries, list the library names (matching the directory names in the pyg-rocm-build directory) separated by commas:
```bash ./build_and_install_pyg.sh --force-rebuild=pytorch_cluster-1.6.3,pytorch_scatter-2.1.2,pytorch_sparse-0.6.18,pytorch_spline_conv-1.2.2```




  
# Build environment
```
Ubuntu 22.04
PyTorch 2.5.0 for Python 3.9~3.12
```
PyTorch 2.4.1 for Python 3.8
ROCm 6.2.0
```
  
# Current version
```
pytorch_geometric-2.5.3
pytorch_scatter-2.1.2
pytorch_sparse-0.6.18
pytorch_cluster-1.6.3
pytorch_spline_conv-1.2.2
```