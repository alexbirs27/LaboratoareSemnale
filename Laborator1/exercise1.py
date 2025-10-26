import numpy as np
import matplotlib.pyplot as plt

startTime = 0
endTime = 0.03
frequency =  1 / 0.0005 # 2000 Hz

t = np.linspace(startTime, endTime, int(frequency * (endTime - startTime)))

functiaX = lambda t : np.cos(520 * np.pi * t + np.pi / 3)
functiaY = lambda t : np.cos(280 * np.pi * t - np.pi / 3)
functiaZ = lambda t : np.cos(120 * np.pi * t + np.pi / 3)


x = np.vectorize(functiaX)(t)
y = np.vectorize(functiaY)(t)
z = np.vectorize(functiaZ)(t)

fig, axs = plt.subplots(3)
fig.suptitle('Tignale cosinusoidale')
axs[0].plot(t, x)
axs[1].plot(t, y)
axs[2].plot(t, z)
plt.savefig('Laborator1/exercitiu1figura1.pdf')
plt.show()

frequency2 = 200 # 200 Hz
t = np.linspace(startTime, endTime, int(frequency2 * (endTime - startTime)))

x1 = np.vectorize(functiaX)(t)
y1 = np.vectorize(functiaY)(t)
z1 = np.vectorize(functiaZ)(t)

fig, axs = plt.subplots(3)
fig.suptitle('Tignale cosinusoidale - frecventa mica ESANTIONAT')
axs[0].stem(t, x1)
axs[1].stem(t, y1)
axs[2].stem(t, z1)
plt.savefig('Laborator1/exercitiu1figura2.pdf')
plt.show()
