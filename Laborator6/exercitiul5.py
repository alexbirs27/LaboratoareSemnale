import numpy as np
import matplotlib.pyplot as plt

def fereastra_dreptunghiulara(N):
    w = np.zeros(N)
    # Maska 1 mijloc . 0 margini
    start = N // 4
    end = 3 * N // 4
    w[start:end] = 1
    return w

def fereastra_hanning(N):
    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / N))
    return w

# Parametri sinusoida
f = 100  # Hz
A = 1
Nw = 200

# Generare timp si semnal
t = np.linspace(0, Nw/1000, Nw)
x = np.vectorize(lambda t: A * np.sin(2 * np.pi * f * t))(t)

# Ferestre craere
w_drept = fereastra_dreptunghiulara(Nw)
w_hanning = fereastra_hanning(Nw)

# Ferestre aplicare
x_drept = x * w_drept
x_hanning = x * w_hanning

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t, x)
axs[0].set_title('Sinusoida originala')
axs[0].set_xlabel('Timp [s]')

axs[1].plot(t, x_drept)
axs[1].set_title('Sinusoida cu fereastra dreptunghiulara')
axs[1].set_xlabel('Timp [s]')

axs[2].plot(t, x_hanning)
axs[2].set_title('Sinusoida cu fereastra Hanning')
axs[2].set_xlabel('Timp [s]')

plt.tight_layout()
plt.savefig('exercitiul5figura.pdf')
plt.show()
