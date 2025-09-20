import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

def smooth_contour(contour, image_shape, kernel_size=(10, 10), iterations=1, pad=100):
    """
    contour를 마스크화한 후 morphological closing으로 smoothing
    - 처리 안정성을 위해 가장자리 충돌을 피하려고 pad 픽셀만큼 패딩을 주고 작업한 뒤 제거.

    Args:
        contour: 원본 contour (N,1,2) 또는 (N,2)
        image_shape: (height, width) of original image
        kernel_size: closing kernel 크기 (예: (10,10))
        iterations: 연산 반복 횟수
        pad: 패딩 픽셀 수 (기본 100)

    Returns:
        매끄럽게 다듬어진 contour (OpenCV 형식, (N,1,2), int32)
    """
    H, W = image_shape[:2]

    cnt = np.asarray(contour)
    if cnt.ndim == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2:
        pass
    elif cnt.ndim == 2 and cnt.shape[1] == 2:
        cnt = cnt.reshape(-1, 1, 2)
    else:
        raise ValueError("contour shape must be (N,1,2) or (N,2)")
    cnt = cnt.astype(np.int32)

    padded_shape = (H + 2*pad, W + 2*pad)
    mask = np.zeros(padded_shape, dtype=np.uint8)

    cnt_padded = cnt.copy()
    cnt_padded[:, 0, 0] += pad
    cnt_padded[:, 0, 1] += pad

    cv2.drawContours(mask, [cnt_padded], -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    new_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not new_contours:
        return cnt

    best = max(new_contours, key=cv2.contourArea).astype(np.int32)
    best[:, 0, 0] -= pad
    best[:, 0, 1] -= pad

    best[:, 0, 0] = np.clip(best[:, 0, 0], 0, W - 1)
    best[:, 0, 1] = np.clip(best[:, 0, 1], 0, H - 1)

    return best


def find_document_contour(original_image: np.ndarray, edged: np.ndarray, save_dir: str = None, show_all: bool = False):
    """
    엣지 이미지에서 작은 노이즈 컨투어를 마스킹하여 제거한 뒤,
    문서(종이, 명함 등)로 추정되는 큰 사각형 영역의 윤곽선을 검출한다.
    다양한 파라미터 조정을 통해 자동 문서 추출 파이프라인에서 활용할 수 있다.

    Args:
        original_image (np.ndarray): 원본 컬러(BGR) 이미지.
        edged (np.ndarray): 엣지(이진) 이미지. 일반적으로 Canny, Morphology 등 후처리된 결과.
        save_dir (str, optional): 결과 시각화 이미지를 저장할 경로. 미지정 시 저장하지 않음.
        show_all (bool, optional): 중간 처리 결과 및 디버그 이미지를 OpenCV 창으로 표시할지 여부.

    Returns:
        screen_cnt (np.ndarray or None): 4점 근사화된 문서 외곽 컨투어(n,1,2) 또는 None.
        best_contour (np.ndarray or None): 내부적으로 smoothing 및 filtering을 거친 가장 큰 문서 컨투어. 
        contours (list): edged에서 검출된 상위 N개의 원본 컨투어 리스트.

    Example:
        screen_cnt, wild_contour, contours, smooth_contour = find_document_contour(img, canny_img)
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = original_image.shape[:2]
    area_thresh = (min(h, w) * 0.1) ** 2

    # 작은 contour 마스크화
    small_mask = np.zeros_like(edged)
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/small_mask.jpg', small_mask)
    if show_all:
        cv2.imshow('Small Mask', small_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 큰 contour 후보만 상위 8개
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    best_contour, best_screen_cnt, best_contour_area, best_contour_solidity = None, None, None, None

    for c in contours:
        doc_mask = np.zeros_like(edged)
        cv2.drawContours(doc_mask, [c], -1, 255, thickness=cv2.FILLED)
        doc_mask_clean = cv2.subtract(doc_mask, small_mask)
        clean_cnts, _ = cv2.findContours(doc_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not clean_cnts:
            continue
        smoothed = max(clean_cnts, key=cv2.contourArea)
        
        area = cv2.contourArea(smoothed)
        hull = cv2.convexHull(smoothed)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)

        if show_all:
            vis = original_image.copy()
            cv2.drawContours(vis, [c], -1, (0,255,0), 2)
            cv2.drawContours(vis, [approx], -1, (0,0,255), 2)
            cv2.drawContours(vis, [smoothed], -1, (255,0,0), 2)
            text = f"solidity: {solidity:.2f} / {len(approx)} / {int(area)}"
            cv2.putText(vis, text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow('Contour Debug', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(approx) == 4 and solidity >= 0.9:
            if best_contour is None:
                best_contour = smoothed
                best_screen_cnt = approx
                best_contour_area = area
                best_contour_solidity = solidity
            elif area >= best_contour_area * 0.95 and solidity > best_contour_solidity:
                    best_contour = smoothed
                    best_screen_cnt = approx
                    best_contour_solidity = solidity

    if best_screen_cnt is None:
        return None, None, None, None
    if show_all:
        cv2.imshow('Best Contour', cv2.drawContours(original_image.copy(), [best_contour], -1, (255, 0, 0), 2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_dir:
        img_draw = cv2.drawContours(original_image.copy(), [best_screen_cnt], -1, (0,255,0), 2)
        img_draw = cv2.drawContours(img_draw, [best_contour], -1, (255,0,0), 2)
        cv2.imwrite(f'{save_dir}/contour.jpg', img_draw)

    s_contour = smooth_contour(best_contour, (h, w), (100, 100))
    return best_screen_cnt, best_contour, contours, s_contour
