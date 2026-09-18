import numpy as np

print("==========================================================")
print("      TALLER 9: REDES NEURONALES - EL PERCEPTRÓN          ")
print("==========================================================")

# ==========================================================
# PARTE 1: TALLER ANALÍTICO (Calculando el Disparo del Crédito)
# ==========================================================
print("\n--- 1. TALLER ANALÍTICO (Aprobación de Crédito) ---")
"""
Datos del problema:
- Entradas: Ingresos (X1) = 50, Deudas (X2) = 20
- Pesos: W1 = 0.8, W2 = -0.5
- Sesgo (Bias): b = -10

Cálculo de la combinación lineal (Z):
Z = (X1 * W1) + (X2 * W2) + b
Z = (50 * 0.8) + (20 * -0.5) + (-10)
Z = 40 + (-10) - 10 = 20
"""
X1, X2 = 50, 20
W1, W2 = 0.8, -0.5
b = -10
Z_credito = (X1 * W1) + (X2 * W2) + b
print(f"Valor calculado de Z para el cliente: {Z_credito}")

# Función Escalón
salida_credito = 1 if Z_credito >= 0 else 0
print(f"Salida de la neurona: {salida_credito} ({'APROBADO' if salida_credito == 1 else 'RECHAZADO'})")
print("Análisis de W2: Tiene sentido que W2 sea negativo porque las deudas restan capacidad de pago, penalizando el puntaje final del cliente.")


# ==========================================================
# PARTE 2: TALLER DE LABORATORIO (El Perceptrón desde Cero - Compuerta OR)
# ==========================================================
print("\n--- 2. TALLER DE LABORATORIO (Compuerta OR) ---")

# 1. Definir la Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Producto punto + Sesgo (Combinación lineal)
    Z = np.dot(X, W) + b
    salida = funcion_escalon(Z)
    return salida

# 3. Configuración para la Compuerta OR
# Regla OR: Debe arrojar 1 si hay al menos un 1, y 0 solo si es [0, 0].
# Pesos y sesgo ajustados manualmente para resolver la compuerta OR:
pesos_or = np.array([0.6, 0.6])
sesgo_or = -0.4

print(f"Pesos configurados: {pesos_or}, Sesgo: {sesgo_or}\n")

# 4. Evaluando todas las combinaciones posibles de la compuerta OR
casos_prueba = [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]

for entrada in casos_prueba:
    resultado = perceptron(entrada, pesos_or, sesgo_or)
    print(f"Entradas: {entrada} ---> Perceptrón disparó: {resultado}")