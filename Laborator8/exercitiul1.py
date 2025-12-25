import numpy as np
import matplotlib.pyplot as plt

# a
N = 1000
t = np.linspace(0, 10, N)

trend = 0.5 * t**2 + 2 * t + 1

# sezon: doua frecvente
sezon = 5 * np.sin(2 * np.pi * t) + 3 * np.sin(2 * np.pi * 2 * t)

# zgomot normal
zgomot = np.random.normal(0, 3, N)

# seria de timp
y = trend + sezon + zgomot

# afisare
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

axs[0].plot(t, trend)
axs[0].set_title('Trend')

axs[1].plot(t, sezon)
axs[1].set_title('Sezon')

axs[2].plot(t, zgomot)
axs[2].set_title('Zgomot')

axs[3].plot(t, y)
axs[3].set_title('Seria de timp')

plt.tight_layout()
plt.savefig('exercitiul1a.pdf')
plt.show()

# b) Autocorelatie
autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / autocorr[0]

plt.figure()
plt.plot(autocorr[:200])
plt.title('Autocorelatia seriei de timp')
plt.xlabel('Lag')
plt.savefig('exercitiul1b.pdf')
plt.show()

# c) Model AR
p = 100 # p mai mare - invata si trendul

# matricea Y cu valorile anterioare
Y = np.zeros((N - p, p))
for i in range(N - p):
    for j in range(p):
        Y[i, j] = y[i + p - 1 - j]

# rezolvam pentru coeficienti
y_target = y[p:]
Gamma = np.dot(Y.T, Y)
gamma = np.dot(Y.T, y_target)
coef = np.dot(np.linalg.inv(Gamma), gamma)

# predictii iterative din primele p valori
predictii = np.zeros(N)
predictii[:p] = y[:p]

for i in range(p, N):
    val = 0
    for j in range(p):
        val += coef[j] * predictii[i - 1 - j]
    predictii[i] = val

plt.figure()
plt.plot(t, y, label='Original')
plt.plot(t, predictii, label='Predictii AR', linestyle='--')
plt.legend()
plt.title(f'Model AR (p={p})')
plt.savefig('exercitiul1c.pdf')
plt.show()

# d) Hyperparameter tuning pentru p si m
p_vals = [5, 10, 20, 50, 100, 200, 400]
m_vals = [100, 200, 400, 600, 800]

best_error = float('inf')
best_p = 0
best_m = 0

for p_test in p_vals:
    for m_test in m_vals:
        if m_test <= p_test:
            continue

        # antrenam pe primele m valori
        y_train = y[:m_test]

        # matricea Y
        Y_train = np.zeros((m_test - p_test, p_test))
        for i in range(m_test - p_test):
            for j in range(p_test):
                Y_train[i, j] = y_train[i + p_test - 1 - j]

        # coeficienti
        y_target_train = y_train[p_test:]
        Gamma_train = np.dot(Y_train.T, Y_train)
        gamma_train = np.dot(Y_train.T, y_target_train)
        coef_train = np.dot(np.linalg.inv(Gamma_train), gamma_train)

        # testare one-step prediction pe restul datelor
        errors = []
        for i in range(m_test, N):
            if i < p_test:
                continue
            # predictie pentru y[i] folosind y[i-1], y[i-2], ..., y[i-p]
            pred = 0
            for j in range(p_test):
                pred += coef_train[j] * y[i - 1 - j]
            errors.append((y[i] - pred)**2)

        if len(errors) > 0:
            mse = np.mean(errors)
            if mse < best_error:
                best_error = mse
                best_p = p_test
                best_m = m_test

print(f'Cei mai buni parametri:')
print(f'p = {best_p}')
print(f'm = {best_m}')
print(f'MSE = {best_error:.4f}')