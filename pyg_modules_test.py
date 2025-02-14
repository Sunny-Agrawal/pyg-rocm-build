#Very basic function test for pyg modules.

import torch
import torch_scatter
import torch_cluster
import torch_spline_conv

#torch_sparse build failing. Ignoring for now.
#import torch_sparse

# Generate test data
x = torch.rand(5, 5)
index = torch.tensor([0, 1, 2, 3, 4])

try:
    print("Testing torch_scatter...")
    result = torch_scatter.scatter_add(x, index, dim=0)
    print("torch_scatter test passed!", result.shape)
except Exception as e:
    print("torch_scatter test failed:", e)

try:
    print("Testing torch_cluster...")
    result = torch_cluster.knn(x, x, k=2)
    print("torch_cluster test passed!", result)
except Exception as e:
    print("torch_cluster test failed:", e)

try:
    print("Testing torch_spline_conv.spline_basis...")
    pseudo = torch.rand(10, 2)  # Example pseudo-coordinates
    kernel_size = torch.tensor([3, 3])  # Example kernel size
    is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # Open spline indicator
    degree = 2  # Example spline degree
    basis = torch_spline_conv.spline_basis(pseudo, kernel_size, is_open_spline, degree)
    print("torch_spline_conv test passed!", basis)
except Exception as e:
    print("torch_spline_conv test failed:", e)

# Commenting out sparse since we know it doesn't work
# try:
#     print("Testing torch_sparse...")
#     result = torch_sparse.spmm(x, x)
#     print("torch_sparse test passed!", result.shape)
# except Exception as e:
#     print("torch_sparse test failed:", e)

print("All module tests complete.")
