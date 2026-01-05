import argparse
import cv2
import numpy as np

from src.detector_orb import extraer_orb, obtener_coincidencias_filtradas
from src.homografia import estimar_homografia_ransac
from src.ar_overlay import aplicar_overlay_ar
from src.utils import cargar_imagen_color, redimensionar_max

def parsear_argumentos():
    #Parsea argumentos de ejecución
    parser = argparse.ArgumentParser(description="AR con Homografía (ORB + RANSAC)")
    parser.add_argument("--modo", choices=["webcam", "imagen"], default="webcam")
    parser.add_argument("--plantilla", required=True, help="Ruta a imagen plantilla")
    parser.add_argument("--overlay", required=True, help="Ruta a imagen overlay")
    parser.add_argument("--entrada", default=None, help="Ruta a imagen de prueba si modo=imagen")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--max_ancho", type=int, default=960, help="Ancho máximo para acelerar")
    parser.add_argument("--min_inliers", type=int, default=18, help="Mínimo de inliers para aceptar detección")
    parser.add_argument("--ratio", type=float, default=0.75, help="Ratio test para filtrar coincidencias")
    return parser.parse_args()

def preparar_plantilla(ruta_plantilla):
    #Carga plantilla y calcula ORB para usarlo en todo el flujo
    img_plantilla = cargar_imagen_color(ruta_plantilla)
    if img_plantilla is None:
        raise FileNotFoundError(f"No se pudo leer la plantilla: {ruta_plantilla}")

    img_plantilla_gray = cv2.cvtColor(img_plantilla, cv2.COLOR_BGR2GRAY)
    puntos_clave_pl, descriptores_pl = extraer_orb(img_plantilla_gray)

    if descriptores_pl is None or len(puntos_clave_pl) < 10:
        raise RuntimeError("La plantilla no tiene suficientes características.")

    alto_pl, ancho_pl = img_plantilla_gray.shape[:2]
    esquinas_plantilla = np.float32([
        [0, 0],
        [ancho_pl - 1, 0],
        [ancho_pl - 1, alto_pl - 1],
        [0, alto_pl - 1]
    ]).reshape(-1, 1, 2)

    return puntos_clave_pl, descriptores_pl, esquinas_plantilla

def procesar_frame(frame_bgr, puntos_clave_pl, descriptores_pl, esquinas_plantilla, overlay_bgr, args):
    #Procesa un frame: detecta plantilla y aplica AR si hay detección válida
    frame_bgr = redimensionar_max(frame_bgr, args.max_ancho)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    puntos_clave_fr, descriptores_fr = extraer_orb(frame_gray)
    if descriptores_fr is None or len(puntos_clave_fr) < 10:
        return frame_bgr, 0, 0

    coincidencias = obtener_coincidencias_filtradas(descriptores_pl, descriptores_fr, ratio=args.ratio)
    if len(coincidencias) < 8:
        return frame_bgr, 0, len(coincidencias)

    matriz_h, mascara_inliers = estimar_homografia_ransac(puntos_clave_pl, puntos_clave_fr, coincidencias)
    if matriz_h is None or mascara_inliers is None:
        return frame_bgr, 0, len(coincidencias)

    num_inliers = int(mascara_inliers.sum())
    frame_salida = frame_bgr.copy()

    if num_inliers >= args.min_inliers:
        contorno = cv2.perspectiveTransform(esquinas_plantilla, matriz_h)
        cv2.polylines(frame_salida, [np.int32(contorno)], True, (0, 255, 0), 2)
        frame_salida = aplicar_overlay_ar(frame_salida, overlay_bgr, contorno)
        cv2.putText(frame_salida, f"coincidencias:{len(coincidencias)} inliers:{num_inliers} DETECTADO",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame_salida, f"coincidencias:{len(coincidencias)} inliers:{num_inliers} NO_DETECTADO",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame_salida, num_inliers, len(coincidencias)

def main():
    args = parsear_argumentos()

    overlay_bgr = cargar_imagen_color(args.overlay)
    if overlay_bgr is None:
        raise FileNotFoundError(f"No se pudo leer overlay: {args.overlay}")

    puntos_clave_pl, descriptores_pl, esquinas_plantilla = preparar_plantilla(args.plantilla)

    if args.modo == "imagen":
        if not args.entrada:
            raise ValueError("En modo imagen debes pasar --entrada ruta/a/imagen.jpg")

        img_entrada = cargar_imagen_color(args.entrada)
        if img_entrada is None:
            raise FileNotFoundError(f"No se pudo leer entrada: {args.entrada}")

        salida, _, _ = procesar_frame(
            img_entrada,
            puntos_clave_pl,
            descriptores_pl,
            esquinas_plantilla,
            overlay_bgr,
            args
        )

        cv2.imshow("AR Homografia - Imagen", salida)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        salida, _, _ = procesar_frame(
            frame,
            puntos_clave_pl,
            descriptores_pl,
            esquinas_plantilla,
            overlay_bgr,
            args
        )

        cv2.imshow("AR Homografia - Webcam", salida)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
