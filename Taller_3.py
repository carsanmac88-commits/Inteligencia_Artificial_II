import cv2
import numpy as np
import matplotlib
# Configuración para entorno remoto/headless (evita errores gráficos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Solución Analítica: Taller 1 (Función Escalón) ---")
# Sub-matriz original de 3x3
matriz_original = np.array([
    [80, 120, 140],
    [90, 200, 210],
    [50, 130, 250]
], dtype=np.uint8)

T = 135
# 1. Aplicación analítica de la umbralización (T = 135)
# Si I(x,y) >= 135 -> 255, de lo contrario -> 0
matriz_resultante = np.where(matriz_original >= T, 255, 0).astype(np.uint8)
print("Matriz resultante con T = 135:")
print(matriz_resultante)

print("\n2. Justificación del error:")
print("- Objetivo ideal: Aislar valores > 100.")
print("- Con T = 135, los valores 120 y 130 (que son mayores a 100) fueron descartados (convertidos en 0).")
print("- Visualmente, el objeto principal se fragmentará o se encogerá artificialmente, perdiendo detalles importantes que sí cumplían con la condición inicial.")
print("="*60)

print("\n--- Laboratorio: Tratamiento de Ruido y Operaciones Morfológicas ---")

# 1. Cargar imagen en escala de grises o generar una sintética de respaldo si no existe 'documento.jpg'
imagen = cv2.imread('documento.jpg', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("[AVISO] No se encontró 'documento.jpg'. Generando imagen sintética de prueba...")
    # Creamos una imagen sintética con un objeto central y ruido simulado
    imagen = np.full((300, 300), 100, dtype=np.uint8)
    # Objeto principal (cuadrado claro)
    imagen[100:200, 100:200] = 200
    # Ruido de sal (puntos blancos en el fondo) y pimienta (huecos negros dentro)
    np.random.seed(42)
    sal_ruido = np.random.rand(*imagen.shape) < 0.02
    pimienta_ruido = np.random.rand(*imagen.shape) < 0.02
    imagen[sal_ruido] = 255
    imagen[pimienta_ruido] = 0

# 2. Binarización estática con un umbral que introduzca ruido intencional
_, imagen_binaria = cv2.threshold(imagen, 110, 255, cv2.THRESH_BINARY)

# 3. Construcción del Elemento Estructurante (Kernel de 3x3)
kernel = np.ones((3, 3), np.uint8)

# 4. Operación Morfológica de Apertura (Erosión seguida de Dilatación) -> Limpia ruido externo (puntos blancos)
apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel, iterations=1)

# 5. Operación Morfológica de Cierre (Dilatación seguida de Erosión) -> Rellena huecos internos (puntos negros)
cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

print("Procesamiento morfológico finalizado correctamente.")

# 6. Generación y guardado de resultados gráficos (compatible con entorno sin pantalla)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("1. Binarizada (Con Ruido)")
plt.imshow(imagen_binaria, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("2. Apertura (Limpia Fondo)")
plt.imshow(apertura, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Cierre (Rellena Huecos)")
plt.imshow(cierre, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("resultado_segmentacion.png")
print("Gráfica guardada exitosamente como 'resultado_segmentacion.png'.")
print("="*60)