import numpy as np

N = 10

p = np.random.randint(-10, 10, N)
q = np.random.randint(-10, 10, N)

print("p(x):", p)
print("q(x):", q)

# Metoda 1: convolutie directa
r1 = np.convolve(p, q)
print("\nConvolutie:", r1)

# Metoda 2: FFT
fft_size = len(p) + len(q) - 1  # padding ca sa evitam convolutia circulara
#echivalent cu np.pad(p, ...)
P = np.fft.fft(p, n=fft_size)
Q = np.fft.fft(q, n=fft_size)
r2 = np.fft.ifft(P * Q).real
print("FFT:", r2)
