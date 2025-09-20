import cv2
import numpy as np

def adjust_contrast_exposure(image: np.ndarray, contrast: float = 1.0, exposure: float = 0.0) -> np.ndarray:
    """
    이미지의 대비와 노출 조절.

    Args:
        image: 입력 이미지
        contrast: 대비
        exposure: 노출

    Returns:
        보정된 이미지
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=exposure)

def adjust_black_point(image: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    블랙포인트 적용.

    Args:
        image: 입력 이미지
        threshold: 블랙포인트 기준값

    Returns:
        블랙포인트 적용 이미지
    """
    result = image.copy()
    result[result <= threshold] = 0
    return result

def adjust_highlight(image: np.ndarray, increase: int = 100) -> np.ndarray:
    """
    하이라이트 강조.

    Args:
        image: 입력 이미지
        increase: 하이라이트 증가값

    Returns:
        하이라이트가 강조된 이미지
    """
    mask = image > 200
    result = image.copy()
    result[mask] = np.clip(result[mask] + increase, 0, 255)
    return result
