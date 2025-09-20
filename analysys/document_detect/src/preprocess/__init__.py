import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from .lighting import reduce_lighting
from .contrast import adjust_contrast_exposure, adjust_black_point, adjust_highlight
from .resize import resize_image, add_padding
from .realtime_preprocess import realtime_preprocess_image

def opencv2pil(opencv_image: np.ndarray) -> Image.Image:
    """OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환"""
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)

def preprocess_image(
    image: np.ndarray,
    max_size: float = 1000.0,
    padding: int = 0,
    reduce_lighting_: bool = False,
    gray: bool = False,
    contrast: float = 1.0,
    exposure: float = 0.0,
    black_point_threshold: int = None,
    highlight_increase: int = None,
    show_steps: bool = False,
    save_steps: str = None
) -> np.ndarray:
    """
    다양한 옵션을 조합해 입력 이미지 전처리

    Args:
        image (np.ndarray): 입력 BGR 이미지
        max_size (float): 리사이즈 최대 변 길이
        padding (int): reduce_lighting 시 임시 패딩값
        reduce_lighting_ (bool): 조명(그림자) 보정 적용 여부
        gray (bool): 흑백 변환 여부
        contrast (float): 대비 조정 인자
        exposure (float): 밝기 조정 인자
        black_point_threshold (int, optional): 블랙포인트 threshold
        highlight_increase (int, optional): 하이라이트 증폭값
        show_steps (bool): 각 단계 이미지 plt로 시각화
        save_steps (str): 각 단계 이미지를 해당 폴더에 저장

    Returns:
        np.ndarray: 최종 전처리된 이미지
    """
    steps = []

    def _save(step_name, img):
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, f'{step_name}.jpg'), img)

    img = resize_image(image, max_size)
    steps.append(('Resized', img.copy())); _save('resized', img)

    if reduce_lighting_:
        img = add_padding(img, top=padding, bottom=padding, left=padding, right=padding)
        img = reduce_lighting(img)
        if padding > 0:
            h, w = img.shape[:2]
            img = img[padding:h-padding, padding:w-padding]
        steps.append(('Lighting Reduced', img.copy())); _save('lighting', img)

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps.append(('Gray', img.copy())); _save('gray', img)

    img = adjust_contrast_exposure(img, contrast=contrast, exposure=exposure)
    steps.append(('Contrast/Exposure', img.copy())); _save('contrast_exposure', img)

    if black_point_threshold is not None:
        img = adjust_black_point(img, threshold=black_point_threshold)
        steps.append(('Black Point', img.copy())); _save('black_point', img)

    if highlight_increase is not None:
        img = adjust_highlight(img, increase=highlight_increase)
        steps.append(('Highlight', img.copy())); _save('highlight', img)

    if show_steps:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
        if len(steps) == 1:
            axes = [axes]
        for ax, (title, step_img) in zip(axes, steps):
            if len(step_img.shape) == 2:
                ax.imshow(step_img, cmap='gray')
            else:
                ax.imshow(opencv2pil(step_img))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return img