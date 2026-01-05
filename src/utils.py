import cv2

def cargar_imagen_color(ruta):
    #Carga una imagen a color en formato BGR
    return cv2.imread(ruta, cv2.IMREAD_COLOR)

def redimensionar_max(img, max_ancho):
    #Redimensiona manteniendo proporción si excede un ancho máximo
    alto, ancho = img.shape[:2]
    if ancho <= max_ancho:
        return img

    escala = max_ancho / float(ancho)
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    return cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
