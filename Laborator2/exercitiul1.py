import numpy as np
import matplotlib.pyplot as plt


startTime = 0
endTime = 5

t = np.linspace(startTime, endTime, 10000)
functiaSin = lambda t : np.sin(10 * np.pi * t)
functiaCos = lambda t : np.cos(10 * np.pi * t - np.pi / 2)

Sin = np.vectorize(functiaSin)(t)
Cos = np.vectorize(functiaCos)(t)

fig, axs = plt.subplots(2)
fig.suptitle("Tignale asemanatoare")
axs[0].plot(t, Sin)
axs[1].plot(t, Cos)
plt.savefig('Laborator2/exercitiul1figura1.pdf')
plt.show()