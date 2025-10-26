import numpy as np
import matplotlib.pyplot as plt

# Ex 2 a:

startTime = 0
endTime = 0.03
nrEsantioane = 1600
functiaX = lambda t : np.cos(800 * np.pi * t + np.pi / 3) # 400 Hz - 800 / 2
t = np.linspace(startTime, endTime, nrEsantioane)
x = np.vectorize(functiaX)(t)

plt.stem(t, x)
plt.plot(t, x)
plt.savefig('Laborator1/exercitiu2figura1.pdf')
plt.show()

# Ex 2 b:
startTime = 0 
endTime = 3
functiaY = lambda t : np.sin(1600 * np.pi * t) # 800 Hz - 1600 / 2
t = np.linspace(startTime, endTime, 10000)
x = np.vectorize(functiaY)(t)

plt.stem(t, x)
plt.plot(t, x)
plt.savefig('Laborator1/exercitiu2figura2.pdf')
plt.show()

# Ex 2 c:
startTime = 0
endTime = 3

functiaZ = lambda t: 2 * np.mod(240 * t, 1) - 1.0 
t = np.linspace(startTime, endTime, 1000)
x = np.vectorize(functiaZ)(t)
plt.plot(t, x)
plt.savefig('Laborator1/exercitiu2figura3.pdf')
plt.show()

# Ex 2 d:
functiaD = lambda t: np.sign(np.sin(600 * np.pi * t))
t = np.linspace(startTime, endTime, 2000)
x = np.vectorize(functiaD)(t)
plt.stem(t, x)
plt.plot(t, x)
plt.savefig('Laborator1/exercitiu2figura4.pdf')
plt.show()

# Ex 2 e:
x = np.random.rand(128, 128)
plt.imshow(x)
plt.savefig('Laborator1/exercitiu2figura5.pdf')
plt.show()

# Ex 2 f:
x = np.linspace(0, 1, 128 * 128)
x = np.reshape(x, (128, 128))
print(x[0, :])
plt.imshow(x)
plt.savefig('Laborator1/exercitiu2figura6.pdf')
plt.show()