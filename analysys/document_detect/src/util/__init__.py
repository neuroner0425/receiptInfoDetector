import cv2
import numpy as np

def show_side_by_side(left_img, right_img, window_name="Compare", save_path=None):
    # 높이 기준으로 사이즈 맞추기
    hL, wL = left_img.shape[:2]
    hR, wR = right_img.shape[:2]
    if hL != hR:
        scale = hL / hR
        right_img = cv2.resize(right_img, (int(wR*scale), hL), interpolation=cv2.INTER_AREA)

    # 가로로 이어 붙이기
    canvas = np.hstack([left_img, right_img])

    # 보기
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 저장 옵션
    if save_path is not None:
        cv2.imwrite(save_path, canvas)