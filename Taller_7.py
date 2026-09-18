import numpy as np
from sklearn.neighbors import KNeighborsClassifier

print("==========================================================")
print("       TALLER 7: KNN - LA VOTACIÓN ESPACIAL Y LAB         ")
print("==========================================================")

# ==========================================================
# PARTE 1: TALLER ANALÍTICO (La Votación Espacial)
# ==========================================================
print("\n--- 1. TALLER ANALÍTICO (Puntos base y Distancia Euclidiana) ---")

# Dataset inicial de la Parte 1
# Caracteristicas: [Edad, Salario en miles]
X_analitico = np.array([
    [20, 30],  # Punto A -> NO COMPRA (0)
    [40, 50],  # Punto B -> COMPRA (1)
    [35, 45]   # Punto C -> COMPRA (1)
])
Y_analitico = np.array([0, 1, 1])

# Punto nuevo a clasificar
punto_nuevo = np.array([[30, 40]])

# Cálculo matemático explícito de la Distancia Euclidiana
print("Calculando Distancias Euclidianas desde el Punto Nuevo (30, 40):")
puntos_nombres = ['A(20, 30)', 'B(40, 50)', 'C(35, 45)']
distancias = []

for i, p in enumerate(X_analitico):
    # Fórmula: d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    dist = np.sqrt(np.sum((punto_nuevo[0] - p) ** 2))
    distancias.append(dist)
    print(f" * Distancia al Punto {puntos_nombres[i]}: {dist:.2f}")

# Clasificación con K=1
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_analitico, Y_analitico)
pred_k1 = knn_k1.predict(punto_nuevo)[0]
print(f"\nResultado con K = 1 -> Clase predicha: {pred_k1} ({'COMPRA' if pred_k1 == 1 else 'NO COMPRA'})")

# Clasificación con K=3
knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k3.fit(X_analitico, Y_analitico)
pred_k3 = knn_k3.predict(punto_nuevo)[0]
print(f"Resultado con K = 3 -> Clase predicha: {pred_k3} ({'COMPRA' if pred_k3 == 1 else 'NO COMPRA'})")
print("  (Nota: Con K=3 votan los 3 vecinos: A=0, B=1, C=1. Por mayoría democrática gana COMPRA).")


# ==========================================================
# PARTE 2: TALLER DE LABORATORIO (Clasificador Universal)
# ==========================================================
print("\n--- 2. TALLER DE LABORATORIO (Dataset Ampliado) ---")

# Dataset de entrenamiento ampliado con 10 puntos y 3 dimensiones: [Edad, Salario, Número de Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # A
    [40, 50, 2],  # B
    [35, 45, 1],  # C
    [25, 35, 0],  # D
    [50, 70, 3],  # E
    [28, 40, 1],  # F
    [45, 60, 2],  # G
    [22, 28, 0],  # H
    [38, 48, 2],  # I
    [60, 80, 4]   # J
])

# Etiquetas correspondientes: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

# Nuevo cliente con 3 características [Edad, Salario, Hijos]
nuevo_cliente_lab = np.array([[30, 40, 1]])

# Experimento con K = 1
modelo_lab_1 = KNeighborsClassifier(n_neighbors=1)
modelo_lab_1.fit(X_entrenamiento, Y_entrenamiento)
pred_lab_1 = modelo_lab_1.predict(nuevo_cliente_lab)[0]
print(f"Laboratorio - Predicción con n_neighbors=1: {pred_lab_1} ({'COMPRA' if pred_lab_1 == 1 else 'NO COMPRA'})")

# Experimento con K = 5
modelo_lab_5 = KNeighborsClassifier(n_neighbors=5)
modelo_lab_5.fit(X_entrenamiento, Y_entrenamiento)
pred_lab_5 = modelo_lab_5.predict(nuevo_cliente_lab)[0]
print(f"Laboratorio - Predicción con n_neighbors=5: {pred_lab_5} ({'COMPRA' if pred_lab_5 == 1 else 'NO COMPRA'})")


# ==========================================================
# PREGUNTA DE ANÁLISIS: LA MALDICIÓN DE LA DIMENSIONALIDAD
# ==========================================================
"""
Pregunta: Si en lugar de 3 columnas tuvieran 1,000 columnas (como los píxeles de una imagen), 
¿qué pasaría matemáticamente con la Distancia Euclidiana entre los puntos?

Respuesta analítica:
A medida que aumentamos las dimensiones a 1,000:
1. Las distancias entre los puntos tienden a concentrarse; la diferencia entre el punto 
   más cercano y el más lejano se vuelve insignificante (pierde contraste espacial).
2. El concepto de "vecindad" colapsa porque todos los puntos comienzan a estar 
   equidistantemente lejos unos de otros en espacios de alta dimensionalidad.
3. El algoritmo KNN deja de ser efectivo y requiere la aplicación previa de algoritmos 
   de reducción de dimensionalidad (como PCA) para aislar las características verdaderamente útiles.
"""