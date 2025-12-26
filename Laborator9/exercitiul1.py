import numpy as np
import matplotlib.pyplot as plt

# 1. Generare serie de timp (din Lab8)
N = 1000
t = np.linspace(0, 10, N)

trend = 0.5 * t**2 + 2 * t + 1
sezon = 5 * np.sin(2 * np.pi * t) + 3 * np.sin(2 * np.pi * 2 * t)
zgomot = np.random.normal(0, 3, N)

y = trend + sezon + zgomot

plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.title('Seria de timp generata')
plt.savefig('exercitiul1.pdf')
plt.show()
