import cv2

def extraer_orb(img_gray, nfeatures=1500):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptores = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptores

def obtener_matches_filtrados(des_pl, des_fr, ratio=0.75):
    """
    BFMatcher + Ratio Test (Lowe)
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des_pl, des_fr, k=2)

    buenos = []
    for m_n in matches_knn:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                buenos.append(m)
    return buenos
