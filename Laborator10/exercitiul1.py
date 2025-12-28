import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 10, N)


trend = 0.7 * t**2 + 2.5 * t + 8
sezon = 4 * np.sin(6 * np.pi * t) + 2.5 * np.cos(12 * np.pi * t)
zgomot = np.random.normal(0, 2.5, N)
y = trend + sezon + zgomot


plt.plot(t, y)
plt.title('Seria de timp generata')
plt.savefig('exercitiul1.pdf')
plt.show()
