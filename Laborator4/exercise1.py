import numpy as np
import matplotlib.pyplot as plt
import time


def compute_dft(input_signal):
    signal_length = len(input_signal)

    # Create DFT transformation matrix
    dft_matrix = np.zeros((signal_length, signal_length), dtype=np.complex128)

    for row_index in range(signal_length):
        for column_index in range(signal_length):
            phase_angle = -2 * np.pi * row_index * column_index / signal_length
            dft_matrix[row_index, column_index] = np.exp(1j * phase_angle)

    # Apply transformation
    fr_domain = np.dot(dft_matrix, input_signal)

    return fr_domain


def compute_fft(input_signal):
    signal_length = len(input_signal)

    # Base case for recursion
    if signal_length == 1:
        return input_signal

    # even and odd samples
    even_samples = compute_fft(input_signal[::2])
    odd_samples = compute_fft(input_signal[1::2])

    factors = np.exp(-2j * np.pi * np.arange(signal_length) / signal_length)

    # Combine results
    half_length = signal_length // 2
    fr = np.concatenate([
        even_samples + factors[:half_length] * odd_samples,
        even_samples + factors[half_length:] * odd_samples
    ])

    return fr


def generate_test_signal(time_samples): #new aproach
    component1 = 3 * np.sin(2 * np.pi * 7 * time_samples)
    component2 = 7 * np.sin(2 * np.pi * 12 * time_samples)
    component3 = 4 * np.sin(2 * np.pi * 20 * time_samples)
    return component1 + component2 + component3

execution_time_dft = []
execution_time_fft = []
execution_time_numpy = []
test_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]



for current_size in test_sizes:
    print(f"Processing size: {current_size}")

    base_frequency = 1
    time_samples = np.linspace(0, 1/base_frequency, current_size + 1)[:-1]
    test_signal = generate_test_signal(time_samples)

    # masure DFT execution time
    time_start_dft = time.perf_counter()
    result_dft = compute_dft(test_signal)
    time_end_dft = time.perf_counter()
    elapsed_dft = (time_end_dft - time_start_dft) * 1000  
    execution_time_dft.append(elapsed_dft)
    print(f"  DFT: {elapsed_dft:.4f} ms")

    # measure FFT execution time
    time_start_fft = time.perf_counter()
    result_fft = compute_fft(test_signal)
    time_end_fft = time.perf_counter()
    elapsed_fft = (time_end_fft - time_start_fft) * 1000
    execution_time_fft.append(elapsed_fft)
    print(f"  FFT: {elapsed_fft:.4f} ms")

    # measure NumPy FFT execution time
    time_start_numpy = time.perf_counter()
    result_numpy = np.fft.fft(test_signal)
    time_end_numpy = time.perf_counter()
    elapsed_numpy = (time_end_numpy - time_start_numpy) * 1000
    execution_time_numpy.append(elapsed_numpy)
    print(f"  NumPy FFT: {elapsed_numpy:.4f} ms\n")

plt.figure(figsize=(10, 6))
plt.plot(test_sizes, execution_time_dft, color='red', label='DFT')
plt.plot(test_sizes, execution_time_fft, color='orange', label='FFT')
plt.plot(test_sizes, execution_time_numpy, color='blue', label='Time')

plt.xlabel('Signal Size (samples)')
plt.ylabel('Execution Time (ms)')
plt.title('FFT vs DFT Performance Comparison')
plt.yscale('log')
plt.grid(True)
plt.tight_layout()

plt.savefig('Laborator4/exercitiul1figura1.pdf')
plt.show()