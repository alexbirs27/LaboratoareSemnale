import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Generare serie de timp
N = 1000
t = np.linspace(0, 10, N)

trend = 0.7 * t**2 + 2.5 * t + 8
sezon = 4 * np.sin(6 * np.pi * t) + 2.5 * np.cos(12 * np.pi * t)
zgomot = np.random.normal(0, 2.5, N)

serie = trend + sezon + zgomot

# Model MA cu orizont q
q = 2


# Media initiala
mu_start = np.mean(serie)

# Definesc bounds pentru parametri: mu si q coeficienti theta
bounds = [(mu_start - 10, mu_start + 10)]  # bounds pentru mu
for _ in range(q):
    bounds.append((-1.0, 1.0))  # bounds pentru theta

# Functie obiectiv care calculeaza MSE
def cost(parametri):
    m = parametri[0]
    coeffs = parametri[1:]

    errs = np.zeros(N)
    total = 0.0

    for idx in range(N):
        estimate = m
        for j in range(q):
            prev_idx = idx - 1 - j
            if prev_idx >= 0:
                estimate += coeffs[j] * errs[prev_idx]

        errs[idx] = serie[idx] - estimate
        total += errs[idx]**2

    return total

# Optimizare globala cu differential evolution
result = differential_evolution(cost, bounds, seed=42, maxiter=300)

mu = result.x[0]
theta = result.x[1:]

print(f'Parametrii MA(q={q}):')
print(f'mu = {mu:.4f}')
print(f'theta = {theta}')

# Generare predictii cu parametrii optimi
epsilon = np.zeros(N)
y_pred = np.zeros(N)

for idx in range(N):
    val_pred = mu
    for j in range(q):
        prev_idx = idx - 1 - j
        if prev_idx >= 0:
            val_pred += theta[j] * epsilon[prev_idx]

    y_pred[idx] = val_pred
    epsilon[idx] = serie[idx] - val_pred

# Vizualizare
plt.subplot(2, 1, 1)
plt.plot(t, serie, label='Serie originala')
plt.title('Serie de timp originala')

plt.subplot(2, 1, 2)
plt.plot(t, serie, label='Original', alpha=0.5)
plt.plot(t, y_pred, label=f'MA(q={q})', color='red')
plt.title(f'Model MA cu q={q}')

plt.tight_layout()
plt.savefig('exercitiul3.pdf')
plt.show()
