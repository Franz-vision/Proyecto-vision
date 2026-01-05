import cv2
import numpy as np

def aplicar_overlay_ar(frame_bgr, overlay_bgr, contorno_4_puntos):
    #Proyecta el overlay sobre el cuadrilátero detectado en el frame usando homografía
    destino = np.float32(contorno_4_puntos.reshape(4, 2))

    alto_ov, ancho_ov = overlay_bgr.shape[:2]
    origen = np.float32([
        [0, 0],
        [ancho_ov - 1, 0],
        [ancho_ov - 1, alto_ov - 1],
        [0, alto_ov - 1]
    ])

    matriz_ov, _ = cv2.findHomography(origen, destino)
    if matriz_ov is None:
        return frame_bgr

    overlay_warp = cv2.warpPerspective(
        overlay_bgr,
        matriz_ov,
        (frame_bgr.shape[1], frame_bgr.shape[0])
    )

    mascara = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mascara, np.int32(destino), 255)

    mascara_3 = cv2.merge([mascara, mascara, mascara])
    mascara_inv = cv2.bitwise_not(mascara_3)

    fondo = cv2.bitwise_and(frame_bgr, mascara_inv)
    frente = cv2.bitwise_and(overlay_warp, mascara_3)
    salida = cv2.add(fondo, frente)

    return salida
