import numpy as np

n = 20
d = 7

x = np.zeros(n)
for i in range(n):
    x[i] = np.random.randint(10,100) * 2 - 3


y = np.zeros(n)
for i in range(n):
    y[i] = x[(i - d) % n]

print("Vector original x:", x)
print("Vector deplasat y:", y)
print("Deplasare aleasa:", d)

# FFT
X = np.fft.fft(x)
Y = np.fft.fft(y)

# Metoda 1:
r1 = np.fft.ifft(X.conj() * Y)
print("\nMetoda 1 - Corelatie:")
print(r1.real)
pozitie1 = np.argmax(np.abs(r1))
print("Deplasare gasita:", pozitie1)

# Metoda 2: 
r2 = np.fft.ifft(Y / X)
print("\nMetoda 2 - Impartire:")
print(r2.real)
pozitie2 = np.argmax(np.abs(r2))
print("Deplasare gasita:", pozitie2)
#metoda 2 nu e prea corecta - impartire la 0