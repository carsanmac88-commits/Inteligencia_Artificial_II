import cv2
import numpy as np
import matplotlib
# Configuración para entorno remoto/headless (evita errores gráficos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Solución Analítica: Bounding Box ---")
# Coordenadas de las esquinas del objeto irregular: A(2,4), B(8,2), C(10,7), D(3,9)
puntos = np.array([[2, 4], [8, 2], [10, 7], [3, 9]])

x_coords = puntos[:, 0]
y_coords = puntos[:, 1]

x_min = np.min(x_coords)
x_max = np.max(x_coords)
y_min = np.min(y_coords)
y_max = np.max(y_coords)

w = x_max - x_min
h = y_max - y_min

print(f"1. Coordenadas extremas del Bounding Box:")
print(f"   - X_min = {x_min}, X_max = {x_max}")
print(f"   - Y_min = {y_min}, Y_max = {y_max}")
print(f"2. Dimensiones de la caja delimitadora:")
print(f"   - Ancho (W) = {w}")
print(f"   - Alto (H) = {h}")
print("="*60)

print("\n--- Laboratorio: Clasificador de Formas y Contornos ---")

# 1. Cargar imagen de prueba o generar una sintética con objetos de diferentes tamaños (monedas/formas)
imagen_color = cv2.imread('objetos.jpg')
if imagen_color is None:
    print("[AVISO] No se encontró 'objetos.jpg'. Generando imagen sintética de prueba con múltiples objetos...")
    imagen_color = np.zeros((400, 400, 3), dtype=np.uint8)
    # Dibujar objetos pequeños (círculos o cuadrados) y grandes
    cv2.circle(imagen_color, (80, 80), 25, (255, 255, 255), -1)    # Objeto pequeño
    cv2.rectangle(imagen_color, (200, 80), (320, 180), (255, 255, 255), -1) # Objeto grande
    cv2.circle(imagen_color, (100, 300), 40, (255, 255, 255), -1)   # Objeto mediano/grande

# 2. Pipeline completo: Conversión a grises -> Umbralización -> Limpieza Morfológica -> Contornos
gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)

# Limpieza morfológica para eliminar ruido
kernel = np.ones((3, 3), np.uint8)
binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# Detección de Contornos (RETR_EXTERNAL ignora agujeros internos)
contornos, _ = cv2.findContours(binaria_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Se detectaron {len(contornos)} contornos candidatos en la imagen.")

# Umbral de área para la lógica empresarial (separar objetos grandes de pequeños)
UMBRAL_AREA = 3000

for i, cnt in enumerate(contornos):
    # Calcular el Área del contorno[cite: 4]
    area = cv2.contourArea(cnt)
    
    # Filtrar ruido menor
    if area > 100:
        print(f" - Objeto {i+1}: Área = {area:.2f} píxeles")
        
        # Calcular el Bounding Box (Rectángulo envolvente)[cite: 4]
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Lógica empresarial: Si el área es mayor al umbral -> Azul (Grande), menor -> Rojo (Pequeño)[cite: 4]
        if area > UMBRAL_AREA:
            color_box = (255, 0, 0) # Azul en BGR
            etiqueta = "Grande"
        else:
            color_box = (0, 0, 255) # Rojo en BGR
            etiqueta = "Pequeno"
            
        # Dibujar rectángulo en la imagen a color[cite: 4]
        cv2.rectangle(imagen_color, (x, y), (x + bw, y + bh), color_box, 2)
        
        # Calcular el centroide (momentos espaciales)[cite: 4]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Dibujar un punto central y texto
            cv2.circle(imagen_color, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(imagen_color, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

print("Procesamiento de extracción de características finalizado correctamente.")

# Guardar resultado gráfico final
cv2.imwrite("resultado_clasificador.png", imagen_color)
print("Imagen de clasificación guardada exitosamente como 'resultado_clasificador.png'.")
print("="*60)