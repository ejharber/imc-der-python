import numpy as np

# Generate three random 3D vectors
# np.random.seed(42)  # For consistent results
v1 = np.random.rand(3)
v2 = np.random.rand(3)
v3 = np.random.rand(3)

# Function to project vector a onto vector b
def project(a, b):
    return (np.dot(a, b) / np.dot(b, b)) * b

# Projection of v1 and v2 onto v3
proj_v1_on_v3 = project(v1, v3)
proj_v2_on_v3 = project(v2, v3)

# Subtract projections
diff_projections = proj_v1_on_v3 - proj_v2_on_v3

# Projection of (v1 - v2) onto v3
proj_diff_on_v3 = project(v1 - v2, v3)

# Check if the two results are equal
are_equal = np.allclose(diff_projections, proj_diff_on_v3)

print(diff_projections, proj_diff_on_v3)

v1, v2, v3, proj_v1_on_v3, proj_v2_on_v3, diff_projections, proj_diff_on_v3, are_equal
