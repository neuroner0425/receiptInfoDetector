import cv2, numpy as np, os

# ---------- helpers ----------
def _order_quad(quad):
    pts = quad.reshape(-1,2).astype(np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.stack([tl,tr,br,bl]).astype(np.float32)

def _rect_size(q):
    tl,tr,br,bl = q
    w = int(round(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))))
    h = int(round(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))))
    return max(w,20), max(h,20)

def _rescale_pts(pts, from_hw, to_hw):
    fh, fw = from_hw; th, tw = to_hw
    P = pts.reshape(-1,2).astype(np.float32).copy()
    P[:,0] *= (tw/float(fw)); P[:,1] *= (th/float(fh))
    return P.reshape(-1,1,2).astype(np.float32)

# ---------- main: Perspective -> TPS (remap) ----------
def rectify_document(
    original_image: np.ndarray,
    screen_cnt: np.ndarray,
    s_contour: np.ndarray,
    contour_shape: tuple
):
    Himg, Wimg = original_image.shape[:2]
    if contour_shape is not None and (contour_shape[0]!=Himg or contour_shape[1]!=Wimg):
        screen_cnt = _rescale_pts(screen_cnt, contour_shape, (Himg, Wimg))
        s_contour  = _rescale_pts(s_contour,  contour_shape, (Himg, Wimg))
    quad = _order_quad(screen_cnt)
    Wt, Ht = _rect_size(quad)
    dst_rect = np.array([[0,0],[Wt-1,0],[Wt-1,Ht-1],[0,Ht-1]], np.float32)

    Hmat = cv2.getPerspectiveTransform(quad, dst_rect)
    warped = cv2.warpPerspective(original_image, Hmat, (Wt, Ht), flags=cv2.INTER_LINEAR)

    return warped