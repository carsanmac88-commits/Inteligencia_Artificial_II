import cv2
import numpy as np
import matplotlib
# Configuración para entorno remoto/headless (evita errores gráficos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Solución Analítica: Taller de Convolución ---")
# 1. Matriz de la sección de imagen (I) de 3x3 proporcionada en el documento
imagen_matriz = np.array([
    [10, 20, 30],
    [15, 250, 15],
    [20, 10, 20]
], dtype=np.float32)

# Kernel de filtro de media (1/9 en cada celda)
# Producto punto y sumatoria para el píxel central (que vale 250)
suma_total = np.sum(imagen_matriz * (1.0 / 9.0))
nuevo_valor_central = round(suma_total, 2)

print(f"Suma de los elementos: 10 + 20 + 30 + 15 + 250 + 15 + 20 + 10 + 20 = {np.sum(imagen_matriz)}")
print(f"Cálculo matemático: {np.sum(imagen_matriz)} / 9 = {suma_total:.2f}")
print(f"Nuevo valor calculado para el píxel central: {nuevo_valor_central}")

print("\n2. Análisis del resultado:")
print("- El valor original 250 era un píxel anómalo ('ruido de sal') en medio de vecinos con intensidades bajas.")
print("- Al aplicar el filtro de media, el valor cae drásticamente a cerca de 22.22 porque se promedia con todo su entorno.")
print("- El filtro de media difumina la imagen porque redistribuye la energía del píxel extremo entre sus vecinos, generando un efecto de mancha o suavizado generalizado.")
print("="*60)

print("\n--- Laboratorio: Estrategias de Suavizado (Filtros Avanzados) ---")

# Cargar imagen o generar una sintética de respaldo con ruido de sal y pimienta
imagen = cv2.imread('imagen_ruidosa.jpg', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("[AVISO] No se encontró 'imagen_ruidosa.jpg'. Generando imagen sintética con ruido de Sal y Pimienta...")
    imagen = np.full((300, 300), 120, dtype=np.uint8)
    # Dibujar una forma geométrica clara (cuadrado central)
    imagen[100:200, 100:200] = 200
    
    # Inyectar ruido de Sal y Pimienta agresivo
    np.random.seed(100)
    sal = np.random.rand(*imagen.shape) < 0.05
    pimienta = np.random.rand(*imagen.shape) < 0.05
    imagen[sal] = 255
    imagen[pimienta] = 0

# Tamaño de Kernel agresivo (7x7) solicitado en el laboratorio
k_size = (7, 7)

# 1. Filtro de Media (Promedio simple)
blur_media = cv2.blur(imagen, k_size)

# 2. Filtro Gaussiano
blur_gauss = cv2.GaussianBlur(imagen, k_size, 0)

# 3. Filtro de Mediana (Ideal para ruido de impulso / Sal y Pimienta)
blur_mediana = cv2.medianBlur(imagen, 7)

print("Procesamiento de filtrado espacial finalizado correctamente.")

# Generación y guardado de resultados gráficos (compatible con entorno sin pantalla)
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original (Con Ruido)")
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("1. Filtro de Media (7x7)")
plt.imshow(blur_media, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("2. Filtro Gaussiano (7x7)")
plt.imshow(blur_gauss, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("3. Filtro de Mediana (7x7)")
plt.imshow(blur_mediana, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("resultado_filtros.png")
print("Gráfica comparativa guardada exitosamente como 'resultado_filtros.png'.")
print("="*60)