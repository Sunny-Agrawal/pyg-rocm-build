import torch
import torch_scatter
import traceback

###################
# Torch-Scatter Test Script
###################
#
# Validates that torch_scatter functions run on GPU.
# Prints input & output for easy verification.
# Captures failures but does not terminate.
###################

# We'll assume torch_scatter is installed.
# If your environment is different, modify imports accordingly.

# Attempt to import scatter functions
# (In your environment, these might be accessible via torch.ops.torch_scatter)
try:
    from torch_scatter import scatter_sum, scatter_mul, scatter_mean, scatter_min, scatter_max
    SCATTER_AVAILABLE = True
except ImportError:
    print("\n[ERROR] torch_scatter not found.")
    SCATTER_AVAILABLE = False

# Keep track of which tests pass/fail.
TEST_RESULTS = []

# Utility function for structured print
def print_result(test_name, passed, err=None):
    if passed:
        print(f"✅ {test_name} passed!")
    else:
        print(f"❌ {test_name} FAILED! Error: {err}")
    TEST_RESULTS.append((test_name, passed, err))

# Optional: define a function to check GPU usage
# We'll just confirm that the output tensor is_cuda == True
def is_on_gpu(tensor):
    return tensor.is_cuda


def test_scatter_sum():
    test_name = "scatter_sum test"
    try:
        # Create a small example
        src = torch.tensor([10, 20, 30, 40], dtype=torch.float)
        index = torch.tensor([2, 0, 1, 2])

        # Move to GPU if available
        if torch.cuda.is_available():
            src = src.cuda()
            index = index.cuda()

        out = scatter_sum(src, index, dim=0)

        # We expect out.size(0) to be at least 3
        # since index max is 2
        # we want out = [20, 30, 50]

        expected = torch.tensor([20., 30., 50.])
        if torch.cuda.is_available():
            expected = expected.cuda()

        # Print debug info
        print(f"\n--- {test_name} ---")
        print("src:", src)
        print("index:", index)
        print("out:", out)
        print("expected:", expected)
        print("out on GPU?", is_on_gpu(out))

        # Check correctness
        passed = torch.allclose(out, expected, atol=1e-6)
        # Check GPU usage if cuda available
        if torch.cuda.is_available():
            passed = passed and out.is_cuda

        print_result(test_name, passed)

    except Exception as e:
        print_result(test_name, False, err=traceback.format_exc())


def test_scatter_mul():
    test_name = "scatter_mul test"
    try:
        # src = [2, 2, 3], index = [0, 0, 2]
        src = torch.tensor([2, 2, 3], dtype=torch.float)
        index = torch.tensor([0, 0, 2])

        if torch.cuda.is_available():
            src = src.cuda()
            index = index.cuda()

        out = scatter_mul(src, index, dim=0)

        # index.max() = 2, so out.size(0) >= 3
        # out[0] = 2 * 2 = 4
        # out[1] = 1 (since not assigned?) => but default is presumably 1?
        # out[2] = 3
        # so expected = [4, 1, 3]

        expected = torch.tensor([4., 1., 3.])
        if torch.cuda.is_available():
            expected = expected.cuda()

        print(f"\n--- {test_name} ---")
        print("src:", src)
        print("index:", index)
        print("out:", out)
        print("expected:", expected)
        print("out on GPU?", is_on_gpu(out))

        passed = torch.allclose(out, expected, atol=1e-6)
        if torch.cuda.is_available():
            passed = passed and out.is_cuda

        print_result(test_name, passed)

    except Exception as e:
        print_result(test_name, False, err=traceback.format_exc())


def test_scatter_mean():
    test_name = "scatter_mean test"
    try:
        # src = [10, 20, 30, 40], index = [0, 0, 1, 1]
        # For dim=0
        # out[0] = mean of (10, 20) = 15
        # out[1] = mean of (30, 40) = 35
        src = torch.tensor([10, 20, 30, 40], dtype=torch.float)
        index = torch.tensor([0, 0, 1, 1])

        if torch.cuda.is_available():
            src = src.cuda()
            index = index.cuda()

        out = scatter_mean(src, index, dim=0)
        expected = torch.tensor([15., 35.])
        if torch.cuda.is_available():
            expected = expected.cuda()

        print(f"\n--- {test_name} ---")
        print("src:", src)
        print("index:", index)
        print("out:", out)
        print("expected:", expected)
        print("out on GPU?", is_on_gpu(out))

        passed = torch.allclose(out, expected, atol=1e-6)
        if torch.cuda.is_available():
            passed = passed and out.is_cuda

        print_result(test_name, passed)

    except Exception as e:
        print_result(test_name, False, err=traceback.format_exc())


def test_scatter_min():
    test_name = "scatter_min test"
    try:
        # src = [10, 5, 8, 2], index = [1, 1, 0, 1]
        # out[0] = min of [8]
        # out[1] = min of [10, 5, 2] => 2
        src = torch.tensor([10, 5, 8, 2], dtype=torch.float)
        index = torch.tensor([1, 1, 0, 1])

        if torch.cuda.is_available():
            src = src.cuda()
            index = index.cuda()

        out, arg_out = scatter_min(src, index, dim=0)

        # expected out = [8, 2]
        expected_out = torch.tensor([8., 2.])
        # arg_out => positions: out[0]=8 => came from src idx=2
        # out[1]=2 => came from src idx=3
        expected_arg_out = torch.tensor([2, 3], dtype=torch.long)

        if torch.cuda.is_available():
            expected_out = expected_out.cuda()
            expected_arg_out = expected_arg_out.cuda()

        print(f"\n--- {test_name} ---")
        print("src:", src)
        print("index:", index)
        print("out:", out)
        print("arg_out:", arg_out)
        print("expected_out:", expected_out)
        print("expected_arg_out:", expected_arg_out)
        print("out on GPU?", is_on_gpu(out))
        print("arg_out on GPU?", is_on_gpu(arg_out))

        passed = torch.allclose(out, expected_out, atol=1e-6)
        passed = passed and torch.equal(arg_out, expected_arg_out)
        if torch.cuda.is_available():
            passed = passed and out.is_cuda and arg_out.is_cuda

        print_result(test_name, passed)

    except Exception:
        print_result(test_name, False, err=traceback.format_exc())


def test_scatter_max():
    test_name = "scatter_max test"
    try:
        # src = [3, 10, 8, 12], index = [0, 1, 1, 1]
        # out[0] = max of [3] => 3
        # out[1] = max of [10, 8, 12] => 12
        src = torch.tensor([3, 10, 8, 12], dtype=torch.float)
        index = torch.tensor([0, 1, 1, 1])

        if torch.cuda.is_available():
            src = src.cuda()
            index = index.cuda()

        out, arg_out = scatter_max(src, index, dim=0)

        # expected out = [3, 12]
        expected_out = torch.tensor([3., 12.])
        # arg_out => positions: out[0]=3 => came from src idx=0
        # out[1]=12 => came from src idx=3
        expected_arg_out = torch.tensor([0, 3], dtype=torch.long)

        if torch.cuda.is_available():
            expected_out = expected_out.cuda()
            expected_arg_out = expected_arg_out.cuda()

        print(f"\n--- {test_name} ---")
        print("src:", src)
        print("index:", index)
        print("out:", out)
        print("arg_out:", arg_out)
        print("expected_out:", expected_out)
        print("expected_arg_out:", expected_arg_out)
        print("out on GPU?", is_on_gpu(out))
        print("arg_out on GPU?", is_on_gpu(arg_out))

        passed = torch.allclose(out, expected_out, atol=1e-6)
        passed = passed and torch.equal(arg_out, expected_arg_out)
        if torch.cuda.is_available():
            passed = passed and out.is_cuda and arg_out.is_cuda

        print_result(test_name, passed)

    except Exception:
        print_result(test_name, False, err=traceback.format_exc())


def main():
    # If scatter not available, abort
    if not SCATTER_AVAILABLE:
        print("[ABORT] torch_scatter is not available. Exiting.")
        return

    # Start tests
    test_scatter_sum()
    test_scatter_mul()
    test_scatter_mean()
    test_scatter_min()
    test_scatter_max()

    # Final summary
    print("\n--- FINAL RESULTS ---")
    fails = 0
    for (test_name, passed, err) in TEST_RESULTS:
        if passed:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
            fails += 1

    print(f"\nTotal tests: {len(TEST_RESULTS)}, Failures: {fails}")

if __name__ == "__main__":
    main()
