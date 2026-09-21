import numpy as np

print("=" * 60)
print("TALLER COMPLETO: SESIÓN 12 - REDES NEURONALES MLP")
print("=" * 60)

# -------------------------------------------------------------
# PARTE 1: TALLER ANALÍTICO - CONTANDO PARÁMETROS
# -------------------------------------------------------------
print("\n[Punto 1] Taller Analítico: Conteo de Parámetros")
print("-" * 60)

# Arquitectura descrita: 3 entradas, 1 Capa Oculta (4 neuronas), 1 Capa de Salida (1 neurona)
num_entradas = 3
neuronas_ocultas = 4
neuronas_salida = 1

# 1. Pesos entre Capa de Entrada y Capa Oculta
pesos_entrada_oculta = num_entradas * neuronas_ocultas

# 2. Sesgos en la Capa Oculta (1 por cada neurona oculta)
sesgos_oculta = neuronas_ocultas

# 3. Pesos y Sesgo entre la Capa Oculta y la Capa de Salida
pesos_oculta_salida = neuronas_ocultas * neuronas_salida
sesgos_salida = neuronas_salida

# 4. Total de parámetros entrenables
total_parametros = pesos_entrada_oculta + sesgos_oculta + pesos_oculta_salida + sesgos_salida

print(f"1. Pesos (W1) de Entrada a Oculta: {num_entradas} x {neuronas_ocultas} = {pesos_entrada_oculta}")
print(f"2. Sesgos (b1) en Capa Oculta: {sesgos_oculta}")
print(f"3. Pesos (W2) de Oculta a Salida: {neuronas_ocultas} x {neuronas_salida} = {pesos_oculta_salida}")
print(f"   Sesgo (b2) en Capa de Salida: {sesgos_salida}")
print(f"4. TOTAL DE PARÁMETROS ENTRENABLES: {total_parametros}")


# -------------------------------------------------------------
# PARTE 2: TALLER DE LABORATORIO - PROPAGACIÓN Y RETO BATCH
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("[Punto 2] Taller de Laboratorio: Propagación y Reto Dimensional")
print("=" * 60)

# Función de activación: Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# El Reto Dimensional: Procesamiento de 2 clientes simultáneos (Matriz de 2x3)
X_batch = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# Pesos y sesgos definidos en el taller (coinciden con los 21 parámetros calculados)
W1 = np.array([
    [ 0.1,  0.2,  0.3,  0.4],
    [-0.5,  0.6,  0.7, -0.8],
    [ 0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# PROCESO CAPA OCULTA (Lote)
Z1_batch = np.dot(X_batch, W1) + b1
A1_batch = sigmoide(Z1_batch)

print("\n--- Z1 (Valores puros de la capa oculta - Matriz 2x4) ---")
print(Z1_batch)

print("\n--- A1 (Valores activados con Sigmoide al rango [0, 1]) ---")
print(A1_batch)

# PROCESO CAPA FINAL (Lote)
Z2_batch = np.dot(A1_batch, W2) + b2
Salida_Final_batch = sigmoide(Z2_batch)

print("\n--- Predicciones Finales (Probabilidades para los 2 clientes) ---")
print("Probabilidades:", np.round(Salida_Final_batch, 4))
print("=" * 60)