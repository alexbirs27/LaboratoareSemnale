import numpy as np
import matplotlib.pyplot as plt

B = 8
t = np.linspace(-3, 3, 1000)  
x_t = np.sinc(B * t)**2  # functia originala: sinc^2(Bt)

frecvente_esantionare = [1, 1.5, 2, 4]  # Hz

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, fs in enumerate(frecvente_esantionare):
    Ts = 1 / fs  

    n = np.arange(-int(3/Ts), int(3/Ts)+1)

    # Valori esantionate
    t_esantionat = n * Ts
    x_esantionat = np.sinc(B * t_esantionat)**2

    # xCReconst(t) = sum_n x[n] * sinc((t - n*Ts) / Ts)
    x_reconstructie = np.zeros_like(t)
    for k in range(len(n)):
        x_reconstructie += x_esantionat[k] * np.sinc((t - t_esantionat[k]) / Ts)

    axs[i].plot(t, x_t, 'b-')
    axs[i].stem(t_esantionat, x_esantionat, 'r', markerfmt='ro')
    axs[i].plot(t, x_reconstructie, 'g--')
    axs[i].set_title(f'fs = {fs} Hz')

plt.savefig('exercitiul1figura1.pdf')
plt.show()
