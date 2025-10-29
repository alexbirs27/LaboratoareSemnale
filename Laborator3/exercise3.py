import numpy as np
import matplotlib.pyplot as plt
import math

# Defined composite signal with three frequency components
signal_composite = lambda t: 3 * np.cos(np.pi * 14 * t) + 8 * np.cos(np.pi * 28 * t) + 4 * np.cos(np.pi * 42 * t)


time_vec = np.linspace(0, 1, 2000)
signal_vals = np.vectorize(signal_composite)(time_vec)

# Plot the composite signal
plt.figure(figsize=(10, 4))
plt.plot(time_vec, signal_vals, linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Composite Signal with 3 Frequency Components')
plt.grid(True, alpha=0.3)
plt.savefig(fname='Laborator3/exercitiul3_signal.pdf')
plt.show()

# Create Fourier matrix
N = 32
fourier_mat = np.zeros((N, N), dtype=np.complex128)

for row in range(N):
    for col in range(N):
        exponent = -2 * np.pi * row * col / N
        fourier_mat[row, col] = math.e**(1j * exponent)

# Calculate sampling parameters
base_freq = 7
fs = N * base_freq  # Sampling rate

# Generate sampled signal
time_samples = np.linspace(0, 1 / base_freq, N + 1)[:-1]
sampled_signal = np.vectorize(signal_composite)(time_samples)

# Compute Fourier transform using matrix multiplication
fourier_coeffs = np.dot(fourier_mat, sampled_signal)

# Plot magnitude of Fourier coefficients
plt.figure(figsize=(10, 6))
freq_indices = np.arange(N)
plt.stem(freq_indices, np.absolute(fourier_coeffs))
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.title('Fourier Transform Magnitude')
plt.savefig(fname='Laborator3/exercitiul3_fourier.pdf')
plt.show()