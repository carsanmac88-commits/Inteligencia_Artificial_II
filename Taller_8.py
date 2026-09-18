import numpy as np
from sklearn.svm import SVC

print("==========================================================")
print("     TALLER 8: SVM - MÁQUINAS DE VECTORES DE SOPORTE      ")
print("==========================================================")

# ==========================================================
# PARTE 1: TALLER ANALÍTICO (Dibujando el Margen)
# ==========================================================
print("\n--- 1. TALLER ANALÍTICO (Análisis Teórico) ---")
"""
Pregunta 3 del Taller Analítico:
Si agregamos un nuevo punto de Clase A en la coordenada (1,1), ¿cambiaría la posición de la línea?
Respuesta: 
No cambiaría la posición de la línea óptima (hiperplano). Teóricamente, a las Máquinas de 
Vectores de Soporte (SVM) solo les importan los puntos críticos que están al borde del margen 
(los vectores de soporte). Como el punto (1,1) está muy alejado de la frontera y bien dentro 
del territorio de la Clase A, no afecta el cálculo del margen máximo y es ignorado por el optimizador.
"""
print("Análisis del punto (1,1) en el margen: El hiperplano permanece estático porque el punto es no crítico.")


# ==========================================================
# PARTE 2: TALLER DE LABORATORIO (Fronteras Lineales y No Lineales)
# ==========================================================
print("\n--- 2. TALLER DE LABORATORIO (Implementación Base con Kernel Lineal) ---")

# 1. Dataset base proporcionado en el taller
X_base = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [6, 6],
    [7, 8],
    [8, 7]
])
Y_base = np.array([0, 0, 0, 1, 1, 1])  # 0 = Clase A, 1 = Clase B

# 2. Inicializar SVM con Kernel Lineal
modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(X_base, Y_base)

# 3. Extraer Vectores de Soporte
print("Vectores de Soporte (Kernel Lineal Base):\n", modelo_lineal.support_vectors_)

# 4. Predicción de prueba
nuevo_punto = np.array([[5, 4]])
pred_base = modelo_lineal.predict(nuevo_punto)
print(f"El punto [5,4] pertenece a la clase: {pred_base[0]} ({'Clase B (1)' if pred_base[0] == 1 else 'Clase A (0)'})")


print("\n--- 3. LABORATORIO: ENGAÑANDO AL KERNEL LINEAL (Punto [5,5]) ---")
# Agregamos el punto [5,5] con etiqueta 0 (Clase A) para generar un conflicto de separabilidad lineal
X_modificado = np.vstack([X_base, [5, 5]])
Y_modificado = np.append(Y_base, 0)

# Reentrenar con kernel lineal
modelo_lineal_conflicto = SVC(kernel='linear')
modelo_lineal_conflicto.fit(X_modificado, Y_modificado)
print("Vectores de soporte con el punto [5,5] (Kernel Lineal forzado):\n", modelo_lineal_conflicto.support_vectors_)
print("-> Conclusión: El kernel lineal sufre para trazar una recta perfecta y se ve forzado, cometiendo errores de clasificación.")


print("\n--- 4. LABORATORIO: APLICANDO EL TRUCO DEL KERNEL ('rbf') ---")
# Cambiamos el hiperparámetro de kernel='linear' a kernel='rbf'
modelo_rbf = SVC(kernel='rbf')
modelo_rbf.fit(X_modificado, Y_modificado)

pred_rbf = modelo_rbf.predict(nuevo_punto)
print(f"Predicción del punto [5,4] usando Kernel RBF: {pred_rbf[0]} ({'Clase B (1)' if pred_rbf[0] == 1 else 'Clase A (0)'})")
print("Vectores de soporte (Kernel RBF):\n", modelo_rbf.support_vectors_)


# ==========================================================
# REFLEXIÓN FINAL: ESCENARIOS DEL MUNDO REAL PARA KERNEL RBF
# ==========================================================
"""
Pregunta de Reflexión: ¿En qué escenario del mundo real (ej. medicina o reconocimiento facial) 
cree que un kernel lineal fallaría completamente y se requeriría RBF?

Respuesta de análisis:
1. Diagnóstico Médico (Ej. Detección de tumores benignos vs malignos): Si los valores sanos 
   forman un núcleo central en los exámenes y las células cancerígenas rodean en forma de anillo 
   o dispersión compleja a ese núcleo, una línea recta jamás podrá separarlos. Se requiere RBF 
   para proyectar los datos y envolver el clúster interno.
2. Reconocimiento Facial: Las características geométricas de los rostros en espacios de alta 
   dimensión no son linealmente separables debido a la variación de iluminación y poses; las 
   fronteras de decisión entre identidades suelen ser curvas y cerradas.
"""