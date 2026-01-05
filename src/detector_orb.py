import cv2

def extraer_orb(img_gray, num_caracteristicas=1500):
    #Extrae puntos clave y descriptores ORB
    orb = cv2.ORB_create(nfeatures=num_caracteristicas)
    puntos_clave, descriptores = orb.detectAndCompute(img_gray, None)
    return puntos_clave, descriptores

def obtener_coincidencias_filtradas(des_plantilla, des_frame, ratio=0.75):
    #Calcula coincidencias con BFMatcher y filtra con Ratio Test
    emparejador = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    coincidencias_knn = emparejador.knnMatch(des_plantilla, des_frame, k=2)

    coincidencias_buenas = []
    for par in coincidencias_knn:
        if len(par) != 2:
            continue
        m, n = par
        if m.distance < ratio * n.distance:
            coincidencias_buenas.append(m)

    return coincidencias_buenas
