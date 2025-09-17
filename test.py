print("Hello word")

import numpy as np

# Create an array with singleton dimensions
arr = np.array([[[1, 2, 3]]])
print(f"Original array shape: {arr.shape}")

# Squeeze the array to remove all singleton dimensions
squeezed_arr = np.squeeze(arr)
print(f"Squeezed array shape: {squeezed_arr.shape}")

# Create another array with specific singleton dimensions
arr_with_axis = np.array([[[[1, 2]], [[3, 4]]]])
print(f"Original array with axis shape: {arr_with_axis.shape}")
print(arr_with_axis)

# Squeeze along a specific axis (axis 0 in this case)
squeezed_axis_arr = np.squeeze(arr_with_axis, axis=0)
print(f"Squeezed array along axis 0 shape: {squeezed_axis_arr.shape}")
print(squeezed_axis_arr)