import cv2
import numpy as np

def estimar_homografia_ransac(kps_pl, kps_fr, matches, reproj_thresh=3.0):
    """
    Estima la homografía usando RANSAC.
    Retorna:
      - H: matriz 3x3 o None
      - mascara_inliers: array booleano (len(matches)) o None
    """
    if len(matches) < 4:
        return None, None

    pts_pl = np.float32([kps_pl[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_fr = np.float32([kps_fr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_pl, pts_fr, cv2.RANSAC, reproj_thresh)

    if H is None or mask is None:
        return None, None

    mascara_inliers = mask.astype(bool).reshape(-1)
    return H, mascara_inliers
