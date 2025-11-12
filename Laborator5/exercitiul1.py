import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# a. E din ora in ora
x = pd.read_csv('/home/alex/Desktop/LaboratoareSemnale/Laborator5/Train.csv', sep = ',')

print("Intervalul este de la:", x.iloc[0]['Datetime'], "pana la", x.iloc[-1]['Datetime'] )

# Perioada de esantionare: 1 ora = 3600 secunde
fs = 1 / 3600  # frecventa de esantionare (Hz)
frecventaMaxima = fs / 2  # frecventa Nyquist
print(f"Frecventa de esantionare: {fs} Hz")
print(f"Frecventa maxima (Nyquist): {frecventaMaxima} Hz = 1/{int(1/frecventaMaxima)} Hz")

# d. Transformata Fourier
semnal = x['Count'].values
N = len(semnal)

# Transformata Fourier
X = np.fft.fft(semnal)

# Modulul transformatei
X = abs(X/N)

X = X[:N//2]

# Generare vector de frecvente
f = fs * np.linspace(0, N//2, N//2) / N

# Afisare grafic
plt.plot(f, X)
plt.xlabel('Frecventa (Hz)')
plt.ylabel('Modul')
plt.title('Transformata Fourier')
plt.savefig('Laborator5/exercitiul1figura1.pdf')
plt.show()

# (e) Verificare si eliminare componenta continua
X_complet = np.fft.fft(semnal)
print(f"\n(e) Componenta DC: ", abs(X_complet[0]/N) )

# Eliminare componenta continua
X_complet[0] = 0
semnal_fara_DC = np.fft.ifft(X_complet).real

# Comparatie
fig, axs = plt.subplots(2, figsize=(12, 8))
axs[0].plot(np.arange(N), semnal)
axs[0].set_title('Semnal original')
axs[1].plot(np.arange(N), semnal_fara_DC)
axs[1].set_title('Semnal fara componenta DC')
plt.tight_layout()
plt.show()

# f. primele 4 secvente
indices = np.argsort(X[1:])[-4:][::-1] + 1  # exclude DC

print("Primele 4 frecvente principale:")
for i, idx in enumerate(indices, 1):
    frecv = f[idx]
    perioada_ore = 1 / (frecv * 3600)
    print(f"{i}. f = {frecv:.2e} Hz, Modul = {X[idx]:.2f}, Perioada = {perioada_ore:.1f} ore")

# g. o luna 
start = 888  # 01-10-2012 00:00
o_luna = 31 * 24  
stop = start + o_luna

plt.figure(figsize=(14, 6))
plt.plot(np.arange(start, stop), semnal[start:stop])
plt.xlabel('Esantioane')
plt.ylabel('Nr masini')
plt.title(f'O luna de trafic ({x.iloc[start]["Datetime"]} - {x.iloc[stop-1]["Datetime"]})')
plt.savefig('Laborator5/exercitiul1figura2.pdf')
plt.show()

# h. Determinare data de inceput
print("h. Metoda: Cautam primele 5 zile consecutive cu trafic maxim intre orele 16-18 (zilele lucratoare), urmate de 2 zile cu trafic mai redus (weekend). Identificam astfel primul ciclu saptamanal complet si deducem ziua de start. Neajunsuri: metoda esueaza daca incepem intr-o sarbatoare, weekend sau daca exista evenimente speciale. Acuratetea depinde de regularitatea traficului etc.")



# i. Filtrare frecvente inalte 
X_filtrat = np.fft.fft(semnal)
frecv_taiere = 1 / (6 * 3600)  # frecventa de taiere: 6 ore

for i in range(N):
    freq_curenta = i * fs / N if i < N//2 else (i - N) * fs / N
    if abs(freq_curenta) > frecv_taiere:
        X_filtrat[i] = 0

semnal_filtrat = np.fft.ifft(X_filtrat).real

# Comparatie
fig, axs = plt.subplots(2, figsize=(14, 8))
axs[0].plot(np.arange(start, stop), semnal[start:stop])
axs[0].set_title('Semnal original (o luna)')
axs[0].set_ylabel('Nr masini')

axs[1].plot(np.arange(start, stop), semnal_filtrat[start:stop])
axs[1].set_title('Semnal filtrat - eliminate frecvente > 1/(6h)')
axs[1].set_xlabel('Esantioane')
axs[1].set_ylabel('Nr masini')
plt.savefig('Laborator5/exercitiul1figura3.pdf')
plt.show()

print("i. Filtrare: Am eliminat toate frecventele mai mari decat 1/(6 ore).")
print("Justificare: Pastram doar variatiile lente (zilnice si saptamanale).")
