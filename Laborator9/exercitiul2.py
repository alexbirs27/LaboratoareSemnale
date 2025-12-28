import numpy as np
import matplotlib.pyplot as plt

# Generare serie de timp
N = 1000
t = np.linspace(0, 10, N)

trend = 0.7 * t**2 + 2.5 * t + 8
sezon = 4 * np.sin(6 * np.pi * t) + 2.5 * np.cos(12 * np.pi * t)
zgomot = np.random.normal(0, 2.5, N)

serie = trend + sezon + zgomot

# Exponential smoothing simplu - cautare alpha
alpha_range = np.linspace(0.01, 0.99, 80)
erori = []

for a in alpha_range:
    s = np.zeros(N)
    s[0] = serie[0]

    err = 0
    for i in range(1, N):
        s[i] = a * serie[i] + (1 - a) * s[i-1]
        if i < N - 1:
            err += (s[i] - serie[i+1])**2

    erori.append(err)

best_a = alpha_range[np.argmin(erori)]
print(f'Alpha optim simplu: {best_a:.4f}')

# Aplicam cu alpha optim
s_optim = np.zeros(N)
s_optim[0] = serie[0]
for i in range(1, N):
    s_optim[i] = best_a * serie[i] + (1 - best_a) * s_optim[i-1]

plt.subplot(2, 1, 1)
plt.plot(t, serie, label='Original')
plt.title('Seria originala')

plt.subplot(2, 1, 2)
plt.plot(t, s_optim, label=f'Smoothing alpha={best_a:.3f}', color='red')
plt.title('Exponential smoothing simplu')

plt.tight_layout()
plt.savefig('exercitiul2a.pdf')
plt.show()

# Double exponential smoothing
best_err = float('inf')
best_alpha = 0
best_beta = 0
best_pred = None

for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for b in [0.1, 0.3, 0.5, 0.7, 0.9]:
        level = np.zeros(N)
        trend_val = np.zeros(N)

        level[0] = serie[0]
        trend_val[0] = serie[1] - serie[0]

        err = 0
        for i in range(1, N):
            level[i] = a * serie[i] + (1 - a) * (level[i-1] + trend_val[i-1])
            trend_val[i] = b * (level[i] - level[i-1]) + (1 - b) * trend_val[i-1]

            pred = level[i] + trend_val[i]
            if i < N - 1:
                err += (pred - serie[i+1])**2

        if err < best_err:
            best_err = err
            best_alpha = a
            best_beta = b
            best_pred = level + trend_val

print(f'Double: alpha={best_alpha:.2f}, beta={best_beta:.2f}')

plt.subplot(2, 1, 1)
plt.plot(t, serie, label='Original')
plt.title('Original')

plt.subplot(2, 1, 2)
plt.plot(t, best_pred, color='green')
plt.title(f'Double exp (alpha={best_alpha}, beta={best_beta})')

plt.tight_layout()
plt.savefig('exercitiul2b.pdf')
plt.show()

# Triple exponential smoothing (Holt-Winters)
L = 150  # lungime sezon
a = 0.4
b = 0.3
g = 0.2

level = np.zeros(N)
trend_comp = np.zeros(N)
sezon_comp = np.zeros(N)

level[0] = serie[0]
trend_comp[0] = 0

for i in range(L):
    sezon_comp[i] = serie[i] - level[0]

for i in range(1, N):
    prev_sezon = sezon_comp[i - L] if i >= L else 0

    level[i] = a * (serie[i] - prev_sezon) + (1 - a) * (level[i-1] + trend_comp[i-1])
    trend_comp[i] = b * (level[i] - level[i-1]) + (1 - b) * trend_comp[i-1]
    sezon_comp[i] = g * (serie[i] - level[i] - trend_comp[i-1]) + (1 - g) * prev_sezon

pred_triple = level + trend_comp + sezon_comp

plt.subplot(2, 1, 1)
plt.plot(t, serie)
plt.title('Original')

plt.subplot(2, 1, 2)
plt.plot(t, pred_triple, color='red')
plt.title(f'Triple exp (alpha={a}, beta={b}, gamma={g})')

plt.tight_layout()
plt.savefig('exercitiul2c.pdf')
plt.show()
