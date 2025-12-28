import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 10, N)

trend = 0.7 * t**2 + 2.5 * t + 8
sezon = 4 * np.sin(6 * np.pi * t) + 2.5 * np.cos(12 * np.pi * t)
zgomot = np.random.normal(0, 2.5, N)
y = trend + sezon + zgomot

# Model AR cu orizont p
# y[t] = x1*y[t-1] + x2*y[t-2] + ... + xp*y[t-p]

p = 50

# Construim matricea Y
# Fiecare linie i
Y = np.zeros((N - p, p))
for i in range(N - p):
    for j in range(p):
        Y[i, j] = y[(N - 1) - i - 1 - j]

# Vector target : serie inversata fara primele p elem
y_rev = y[::-1]
y_target = y_rev[:-p]


# Gamma = Y^T * Y
# gamma = Y^T * y
# x_star = inv(Gamma) * gamma

Gamma = np.dot(Y.T, Y)
gamma = np.dot(Y.T, y_target)
x_star = np.dot(np.linalg.inv(Gamma), gamma)

print(f'Model AR(p={p})')
print(f'Coeficienti: {x_star}')

# Generam predictii cu modelul AR
predictions = y[:p].tolist()

for i in range(p, N):
    # Predictia: suma(x_star[j] * predictions[i-1-j] pentru j in 0..p-1)
    pred = np.dot(np.flip(predictions[-p:]), x_star)
    predictions.append(pred)

predictions = np.array(predictions)

# Vizualizare
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, y, label='Serie originala', alpha=0.7)
plt.title('Serie de timp originala')

plt.subplot(2, 1, 2)
plt.plot(t, y, label='Original', alpha=0.5)
plt.plot(t, predictions, label=f'AR(p={p})', color='red', linewidth=1.5)
plt.title(f'Model AR cu orizont p={p}')

plt.tight_layout()
plt.savefig('exercitiul2.pdf')
plt.show()

# Calculam eroarea
mse = np.mean((y - predictions)**2)
print(f'MSE = {mse:.2f}')
