import numpy as np

print("==================================================")
print("  TALLER 1: REPASO DE ÁLGEBRA LINEAL Y PYTHON")
print("  Inteligencia Artificial II - Institución Universitaria de Colombia")
print("==================================================\n")

# =========================================================================
# 1. INDEXACIÓN Y TENSORES
# =========================================================================
print("--- 1. Indexación y Tensores (Escala de Grises y RGB) ---")

# Matriz A simulando imagen en escala de grises (5x5)
Matriz_A = np.array([
    [0, 255, 255, 255, 0],
    [255, 0, 0, 0, 255],
    [255, 0, 128, 0, 255],
    [255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0]
], dtype=np.uint8)

print("Matriz A (5x5):\n", Matriz_A)
print(f"Valor exacto del elemento A[2, 3] (Fila 2, Columna 3): {Matriz_A[2, 3]}")

# Tensor RGB de 1080 x 1920 x 3
M, N, C = 1080, 1920, 3
total_bytes = M * N * C
print(f"Tensor RGB ({M}x{N}x{C}) -> Total de valores en memoria sin comprimir: {total_bytes:,} bytes (~{total_bytes / (1024*1024):.2f} MB)\n")


# =========================================================================
# 2. TRANSFORMACIONES AFINES (BRILLO Y CONTRASTE)
# =========================================================================
print("--- 2. Laboratorio de Transformaciones Afines ---")

# Simulación de una radiografía sobreexpuesta (valores altos)
imagen_radiografia = np.random.randint(200, 255, (5, 5)).astype(np.float32)
print("Matriz Original (Radiografía):\n", imagen_radiografia)

# Reducción de contraste del 50% (alpha = 0.5) y disminución de brillo en 50 unidades (beta = -50)
alpha = 0.5
beta = -50.0
imagen_procesada = alpha * imagen_radiografia + beta

# Acotamiento estricto (Clipping) a [0, 255] y conversión a uint8
imagen_procesada = np.clip(imagen_procesada, 0, 255).astype(np.uint8)
print("Matriz Procesada (Contraste y Brillo ajustados):\n", imagen_procesada, "\n")


# =========================================================================
# 3. TRANSFORMACIONES ESPACIALES, TRANSPOSICIÓN Y APLANAMIENTO
# =========================================================================
print("--- 3. Transposición y Aplanamiento (Flatten) ---")

Matriz_B = np.array([
    [10, 20, 30],
    [40, 50, 60]
], dtype=np.float32)

print("Matriz Original forma:", Matriz_B.shape)
print(Matriz_B)

# Transpuesta (Intercambia filas por columnas)
A_transpuesta = Matriz_B.T
print("\nForma transpuesta (A^T):", A_transpuesta.shape)
print(A_transpuesta)

# Aplanamiento (Vectorización para Redes Neuronales Densas)
vector_1D = Matriz_B.flatten()
print(f"\nVector plano (Tamaño {(200*200*3):,} neuronas aprox. para RGB 200x200x3):")
print("Ejemplo vector 1D obtenido:", vector_1D, "\n")


# =========================================================================
# 4. PRODUCTO HADAMARD VS PRODUCTO PUNTO Y KERNELS DE CONVOLUCIÓN
# =========================================================================
print("--- 4. Laboratorio Final: Programando un Kernel (Convolución Básica) ---")

# Sección de Imagen (I) y Kernel (K) de realce
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
], dtype=np.float32)

K = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Producto Hadamard (elemento a elemento) y suma total de la vecindad
resultado_hadamard = I * K
pixel_central_calculado = np.sum(resultado_hadamard)

print(f"Valor del píxel central calculado mediante convolución local: {pixel_central_calculado}")
print("==================================================")