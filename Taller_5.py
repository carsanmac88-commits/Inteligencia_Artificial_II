import cv2
import numpy as np
import matplotlib
# Configuración para entorno remoto/headless (evita errores gráficos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Solución Analítica: Taller de Gradientes y Sobel ---")
# Matriz de imagen de 3x3 con un borde vertical perfecto (mitad izquierda negra, mitad derecha blanca)
matriz_imagen = np.array([
    [0, 0, 255],
    [0, 0, 255],
    [0, 0, 255]
], dtype=np.float32)

# Kernel Sobel X (Gx)
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Kernel Sobel Y (Gy)[cite: 3]
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 1. Cálculo de la convolución con Sobel X enfocándonos en el píxel central
# Extracción de la región 3x3 centrada (toda la matriz)
# Operación de correlación/convolución cruzada: suma elemento a elemento
grad_x_central = np.sum(matriz_imagen * kernel_sobel_x)
print(f"1. Valor del gradiente en X para el píxel central: {grad_x_central}")
print("   (El valor elevado indica un cambio de intensidad muy fuerte y abrupto en el eje horizontal).")

# 2. Cálculo de la convolución con Sobel Y
grad_y_central = np.sum(matriz_imagen * kernel_sobel_y)
print(f"\n2. Valor del gradiente en Y para el píxel central: {grad_y_central}")
print("   - ¿Por qué el resultado es cero? Porque a lo largo de las filas (eje vertical), los valores no cambian (todas las filas son idénticas: 0, 0, 255).")
print("   - ¿Qué indica sobre la dirección del borde? Indica que el borde es estrictamente vertical; no hay variación de intensidad de arriba hacia abajo, solo de izquierda a derecha.")
print("="*60)

print("\n--- Laboratorio: Inspector de Bordes (Sobel y Canny) ---")

# Cargar imagen o generar una sintética de respaldo con formas geométricas si no existe 'carretera.jpg'
imagen = cv2.imread('carretera.jpg', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("[AVISO] No se encontró 'carretera.jpg'. Generando imagen sintética de prueba con formas geométricas...")
    imagen = np.zeros((300, 300), dtype=np.uint8)
    # Dibujar formas geométricas claras y rectángulos para simular edificios o carriles
    cv2.rectangle(imagen, (50, 50), (150, 250), 255, -1)
    cv2.circle(imagen, (220, 150), 50, 200, -1)
    # Agregar algo de textura o ruido leve
    noise = np.random.randint(0, 50, imagen.shape, dtype=np.uint8)
    imagen = cv2.add(imagen, noise)

# 1. Detección de Bordes con Sobel X y Sobel Y (usando cv2.CV_64F para evitar pérdida de valores negativos)[cite: 3]
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

# Conversión a formato visualizable de 8 bits (valores absolutos)[cite: 3]
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 2. Detección de Bordes con Canny (Umbrales estándar: 50 y 150)[cite: 3]
bordes_canny = cv2.Canny(imagen, 50, 150)

print("Procesamiento de gradientes y detección de bordes finalizado correctamente.")

# 3. Generación del panel de visualización comparativo (Sobel X, Sobel Y y Canny)[cite: 3]
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original / Escala de Grises")
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Sobel X (Bordes Verticales)")
plt.imshow(sobel_x_abs, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Sobel Y (Bordes Horizontales)")
plt.imshow(sobel_y_abs, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Algoritmo Canny (Estándar de Oro)")
plt.imshow(bordes_canny, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("resultado_bordes.png")
print("Panel de visualización guardado exitosamente como 'resultado_bordes.png'.")
print("="*60)