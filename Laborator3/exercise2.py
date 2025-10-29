import numpy as np
import matplotlib.pyplot as plt
import math

signal_func = lambda t: np.sin(10 * np.pi * t + np.pi / 4)
time_points = np.linspace(0, 1, 1000)
signal_values = np.vectorize(signal_func)(time_points)

# Plot the signal in time domain
plt.plot(time_points, signal_values)
plt.xlabel('Time (normalized)')
plt.ylabel('Amplitude')
plt.title('Sinusoidal Signal')
plt.savefig(fname='Laborator3/exercitiul2_signal.pdf')
plt.show()

# Create subplots for different wrapping frequencies
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Test different wrapping frequencies
wrapping_freqs = [2, 3, 4, 6]

for i, omega in enumerate(wrapping_freqs):
    # Calculate complex exponential
    exponential = lambda t: math.e ** (1j * -2 * np.pi * omega * t)
    complex_values = np.vectorize(exponential)(time_points)
    
    # Deform by signal amplitude
    deformed_complex = complex_values * signal_values
    
    # Get current subplot
    current_ax = axs[i // 2, i % 2]
    current_ax.set_xlim(-1.2, 1.2)
    current_ax.set_ylim(-1.2, 1.2)
    current_ax.set_xlabel('Real')
    current_ax.set_ylabel('Imaginary')
    current_ax.set_title(f'Omega = {omega} Hz')
    current_ax.grid(True, alpha=0.3)
    
    # Extract real and imaginary components
    real_parts = [val.real for val in deformed_complex]
    imag_parts = [val.imag for val in deformed_complex]
    
    # Color mapping based on amplitude
    color_values = np.abs(signal_values)
    
    # Animate the drawing process
    step_size = 10
    for index in range(0, len(real_parts), step_size):
        current_ax.scatter(
            real_parts[index:index + step_size],
            imag_parts[index:index + step_size],
            color=plt.cm.plasma(color_values[index:index + step_size]),
            s=1.5
        )
        plt.pause(0.00001)

plt.tight_layout()
plt.savefig(fname='Laborator3/exercitiul2_wrapping.pdf')
plt.show()