import cv2
import numpy as np

def estimar_homografia_ransac(puntos_clave_plantilla, puntos_clave_frame, coincidencias, umbral_reproyeccion=3.0):
    #Estima la homografía usando RANSAC y devuelve la matriz H y la máscara de inliers
    if len(coincidencias) < 4:
        return None, None

    puntos_plantilla = np.float32(
        [puntos_clave_plantilla[m.queryIdx].pt for m in coincidencias]
    ).reshape(-1, 1, 2)

    puntos_frame = np.float32(
        [puntos_clave_frame[m.trainIdx].pt for m in coincidencias]
    ).reshape(-1, 1, 2)

    matriz_homografia, mascara = cv2.findHomography(
        puntos_plantilla,
        puntos_frame,
        cv2.RANSAC,
        umbral_reproyeccion
    )

    if matriz_homografia is None or mascara is None:
        return None, None

    mascara_inliers = mascara.astype(bool).reshape(-1)
    return matriz_homografia, mascara_inliers
