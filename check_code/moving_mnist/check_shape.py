import numpy as np

# List of file paths to check
file_paths = [
    '/workspace/data/org_data/moving_mnist/mnist_test_seq.npy',
    '/workspace/data/preprocessed/bspline_transformed/transformed_global.npy',
    '/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail_global.npy',
    '/workspace/data/preprocessed/bspline_transformed/transformed_cut_global.npy',
    '/workspace/data/preprocessed/bspline_transformed/transformed_global.npy'
]

# Iterate through each file and print its shape and dtype
for file_path in file_paths:
    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}\n")
    except Exception as e:
        print(f"Could not load file {file_path}: {e}\n")