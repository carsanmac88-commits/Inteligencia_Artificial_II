import cv2
import matplotlib.pyplot as plt
import numpy as np

print("==================================================")
print("  TALLER 2: TENSOR DE COLOR Y ANÁLISIS ESTADÍSTICO")
print("  Inteligencia Artificial II - Institución Universitaria de Colombia")
print("==================================================\n")

# =========================================================================
# 1. TALLER ANALÍTICO 1: INDEXACIÓN Y OPERACIONES CON TENSORES
# =========================================================================
print("--- 1. Taller Analítico: Indexación y Slicing Avanzado ---")

# Simulación de un tensor de imagen RGB (Alto: 1080, Ancho: 1920, Canales: 3)
# Creamos una matriz de prueba para ilustrar las dimensiones y el slicing
altura_img, ancho_img = 1080, 1920
tensor_imagen_prueba = np.zeros((altura_img, ancho_img, 3), dtype=np.uint8)

# Aplicando la instrucción de slicing solicitada: recorte = imagen[100:200, 300:400, 1]
recorte = tensor_imagen_prueba[100:200, 300:400, 1]

print(f"Dimensiones de la imagen original: ({altura_img}, {ancho_img}, 3)")
print(f"Instrucción ejecutada: recorte = imagen[100:200, 300:400, 1]")
print(f"Dimensiones exactas (shape) de la variable recorte resultante: {recorte.shape}")
print("Contenido: Matriz 2D de 100x100 píxeles con la información exclusiva del canal verde (índice 1 en BGR).\n")


# =========================================================================
# 2. TALLER DE LABORATORIO 1: TRANSFORMACIÓN DE ESPACIOS (BGR A GRIS)
# =========================================================================
print("--- 2. Laboratorio: Transformación de Píxel BGR a Escala de Grises ---")

# 1. Crear un píxel BGR de prueba completamente amarillo intenso [B, G, R]
pixel = np.array([0, 255, 255], dtype=np.float32)
print(f"Píxel BGR original (Amarillo Puro): {pixel}")

# 2. Calcular matemáticamente su valor en escala de grises usando la fórmula ponderada:
# Y = 0.299 * R + 0.587 * G + 0.114 * B (Ajustado al orden BGR: W = [0.114, 0.587, 0.299])
pesos_bgr = np.array([0.114, 0.587, 0.299])
valor_gris = np.dot(pixel, pesos_bgr)

print(f"Valor de gris calculado matemáticamente: {valor_gris:.2f}")
print(f"Valor entero redondeado (0-255): {int(round(valor_gris))}\n")

# 4. Comprobación práctica usando OpenCV con una imagen local (si existe)
try:
    imagen_cv = cv2.imread('muestra.jpg')
    if imagen_cv is not None:
        img_gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        print("Conversión OpenCV a escala de grises ejecutada exitosamente sobre 'muestra.jpg'.")
    else:
        raise FileNotFoundError
except Exception:
    print("Nota: 'muestra.jpg' no se encuentra en el directorio, omitiendo carga de imagen externa de OpenCV.")


# =========================================================================
# 3. TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO (HISTOGRAMAS)
# =========================================================================
print("\n--- 3. Laboratorio: Análisis Estadístico e Histogramas de Canales ---")

try:
    imagen_rgb = cv2.imread('muestra.jpg')
    if imagen_rgb is None:
        raise FileNotFoundError
except Exception:
    print("Generando imagen sintética RGB de prueba para el cálculo de histogramas...")
    imagen_rgb = np.random.randint(50, 220, (400, 400, 3), dtype=np.uint8)

# Separar la imagen en sus 3 canales individuales (B, G, R)
canal_b = imagen_rgb[:, :, 0]
canal_g = imagen_rgb[:, :, 1]
canal_r = imagen_rgb[:, :, 2]

# Calcular el histograma para cada canal por separado (rangos de 0 a 256)
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

# Graficar los tres histogramas superpuestos utilizando Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist_b, color='blue', label='Canal Azul (B)')
plt.plot(hist_g, color='green', label='Canal Verde (G)')
plt.plot(hist_r, color='red', label='Canal Rojo (R)')
plt.title("Distribución Estadística de Intensidades por Canal (Histograma RGB)")
plt.xlabel("Valor del Píxel (0-255)")
plt.ylabel("Frecuencia (Cantidad de píxeles)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

print("Histogramas procesados correctamente. Desplegando gráfico...")
plt.show()

print("==================================================")