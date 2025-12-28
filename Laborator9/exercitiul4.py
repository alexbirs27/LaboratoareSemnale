import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generare serie de timp
N = 1000
t = np.linspace(0, 10, N)

trend = 0.7 * t**2 + 2.5 * t + 8
sezon = 4 * np.sin(6 * np.pi * t) + 2.5 * np.cos(12 * np.pi * t)
zgomot = np.random.normal(0, 2.5, N)

serie = trend + sezon + zgomot

max_p = 20
max_q = 20

best_aic = float('inf')
best_p = 0
best_q = 0
best_model = None

print('Cautare parametri optimi p si q...')

for p in range(0, max_p + 1, 3):
    for q in range(0, max_q + 1, 3):
        if p == 0 and q == 0:
            continue

        model = ARIMA(serie, order=(p, 0, q))
        model_fit = model.fit()
        aic = model_fit.aic
        
        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_q = q
            best_model = model_fit

print(f'\nParametri optimi:')
print(f'p = {best_p}')
print(f'q = {best_q}')
print(f'AIC = {best_aic:.2f}')

# Pred cu optim
y_pred = best_model.fittedvalues

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, serie, label='Serie originala')
plt.title('Serie de timp originala')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, serie, label='Original', alpha=0.5)
plt.plot(t, y_pred, label=f'ARMA(p={best_p}, q={best_q})', color='red')
plt.title(f'Model ARMA optim')
plt.legend()


plt.tight_layout()
plt.savefig('exercitiul4.pdf')
plt.show()
