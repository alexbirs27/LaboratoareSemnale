import numpy as np
import matplotlib.pyplot as plt
import math

# Matrix size
N = 26

# Initialize the Fourier matrix with complex zeros
fourier_matrix = np.zeros((N, N), dtype=np.complex128)

# Construct the Fourier matrix
for row_idx in range(N):
    for col_idx in range(N):
        exponent = -2 * np.pi * row_idx * col_idx / N # Formula from the course
        fourier_matrix[row_idx, col_idx] = math.e**(1j * exponent) # e^( -j*2pi*row*col/N )

# Plot selected rows of the Fourier matrix
fig, axes = plt.subplots(3, figsize=(10, 12))
time_axis = np.linspace(0, 1, N)

# Plot first 3 rows

for row_number, idx in zip(range(3), range(3)):
    current_row = fourier_matrix[row_number]
    real_part = np.array([element.real for element in current_row])
    imag_part = np.array([element.imag for element in current_row])
    
    axes[idx].set_title(f'Fourier Matrix Row {row_number}')
    axes[idx].plot(time_axis, real_part, label='Real')
    axes[idx].plot(time_axis, imag_part, '--', label='Imaginary')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Laborator3/exercitiul1_fourier_matrix.pdf')
plt.show()

# Verify unitarity of Fourier matrix
F_hermitian = np.matrix(fourier_matrix).getH()
product = F_hermitian * np.matrix(fourier_matrix)

print("\nUnitarity verification:")
print("F^H * F product:")
print(product)

# Check if F^H * F equals N * Identity matrix
identity_check = np.allclose(product, N * np.eye(N), atol=1e-9)
print(f'\nIs F^H * F = N*I? {identity_check}')