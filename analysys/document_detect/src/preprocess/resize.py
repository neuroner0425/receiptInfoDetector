import cv2
import numpy as np

def resize_image(image: np.ndarray, max_size: float = 1000.0) -> np.ndarray:
    """
    이미지 크기 조정.

    Args:
        image: 입력 이미지
        max_size: 최대 한 변 크기

    Returns:
        크기 조정된 이미지
    """
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        ratio = max_size / max(height, width)
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    return image

def add_padding(
    image: np.ndarray, 
    top: int = 10, 
    bottom: int = 10, 
    left: int = 10, 
    right: int = 10, 
    color: tuple = (100, 100, 100)
) -> np.ndarray:
    """
    이미지 주변에 단색 패딩을 추가

    Args:
        image: 입력 이미지 (BGR 또는 GRAY)
        top, bottom, left, right: 각 방향 패딩 크기 (픽셀)
        color: 패딩 색상 (BGR 튜플, 흑백일 땐 정수)

    Returns:
        패딩이 추가된 이미지
    """
    if len(image.shape) == 2:  # GRAY
        color_val = color[0] if isinstance(color, (list, tuple)) else color
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_val)
    else:  # BGR
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded