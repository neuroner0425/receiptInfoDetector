import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

def realtime_edge_dect(image):
    """
    입력 이미지를 대상으로 다양한 전처리 및 엣지 파라미터(contrast, exposure, padding, canny, morph 등)를
    Matplotlib 슬라이더/토글 UI로 실시간 조절하면서, 각 단계(원본, 전처리, 엣지, 컨투어 결과)를 시각적으로 비교할 수 있는 인터랙티브 도구.

    Args:
        image (np.ndarray): 원본 컬러(BGR) 또는 그레이스케일 이미지.

    Returns:
        None. (인터랙티브 GUI를 띄움)

    Note:
        - Matplotlib 기반 인터페이스로, 슬라이더/체크박스 UI 조작 시 자동으로 각 단계 이미지가 갱신됨.
        - 내부적으로 preprocess_image, find_document_contour 등 커스텀 함수와 외부 변환 함수 호출이 필요.

    Example:
        realtime_edge_dect(cv2.imread("document.jpg"))
    """
    
    from .. import preprocess
    from . import contour

    orig_img = image.copy()
    resize_img = preprocess.resize_image(orig_img)

    fig = plt.figure(figsize=(12, 7))
    ax_img1 = fig.add_axes([0.05, 0.55, 0.30, 0.4])
    ax_img2 = fig.add_axes([0.40, 0.55, 0.30, 0.4])
    ax_img3 = fig.add_axes([0.05, 0.05, 0.30, 0.4])
    ax_img4 = fig.add_axes([0.40, 0.05, 0.30, 0.4])

    ax_contrast = fig.add_axes([0.75, 0.88, 0.20, 0.04])
    ax_exposure = fig.add_axes([0.75, 0.83, 0.20, 0.04])
    ax_padding = fig.add_axes([0.75, 0.78, 0.20, 0.04])
    ax_canny1 = fig.add_axes([0.75, 0.73, 0.20, 0.04])
    ax_canny2 = fig.add_axes([0.75, 0.68, 0.20, 0.04])
    ax_closing = fig.add_axes([0.75, 0.63, 0.20, 0.04])
    ax_gradient = fig.add_axes([0.75, 0.58, 0.20, 0.04])
    ax_checkbox = fig.add_axes([0.75, 0.20, 0.20, 0.35])

    slider_contrast = Slider(ax_contrast, 'Contrast', 0.5, 3.0, valinit=1.5, valstep=0.05)
    slider_exposure = Slider(ax_exposure, 'Exposure', -255, 255, valinit=-40, valstep=1)
    slider_padding = Slider(ax_padding, 'Padding', 0, 100, valinit=100, valstep=1)
    slider_canny1 = Slider(ax_canny1, 'Canny th1', 0, 600, valinit=125, valstep=1)
    slider_canny2 = Slider(ax_canny2, 'Canny th2', 0, 600, valinit=150, valstep=1)
    slider_closing = Slider(ax_closing, 'Closing k', 1, 20, valinit=5, valstep=1)
    slider_gradient = Slider(ax_gradient, 'Grad k', 1, 10, valinit=2, valstep=1)

    checkbox = CheckButtons(
        ax_checkbox,
        ['Reduce Lighting', 'Gray', 'Black Point', 'Highlight', 'Use Closing'],
        [True, False, False, False, True]
    )

    def apply_preproc():
        cbs = checkbox.get_status()
        return preprocess.preprocess_image(
            orig_img, max_size=1000.0,
            padding=int(slider_padding.val),
            reduce_lighting_=cbs[0], gray=cbs[1],
            contrast=slider_contrast.val,
            exposure=slider_exposure.val,
            black_point_threshold=100 if cbs[2] else None,
            highlight_increase=100 if cbs[3] else None
        )

    proc_img = apply_preproc()
    ax_img1.set_title("Original");  ax_img1.axis('off')
    ax_img2.set_title("Preprocess"); ax_img2.axis('off')
    ax_img3.set_title("Edge");      ax_img3.axis('off')
    ax_img4.set_title("Contour");   ax_img4.axis('off')

    im_origin = ax_img1.imshow(preprocess.opencv2pil(resize_img))
    im_proc = ax_img2.imshow(proc_img if len(proc_img.shape)==2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB),
                            cmap='gray' if len(proc_img.shape)==2 else None)

    canny_img = cv2.Canny(proc_img, slider_canny1.val, slider_canny2.val)
    k = np.ones((slider_closing.val, slider_closing.val), np.uint8)
    close_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (slider_gradient.val, slider_gradient.val))
    grad_img = cv2.morphologyEx(close_img, cv2.MORPH_GRADIENT, k2)
    im_canny = ax_img3.imshow(grad_img, cmap='gray')

    cnt, best_contour, contours, best_hull = contour.find_document_contour(resize_img, grad_img)
    draw_all = cv2.drawContours(resize_img.copy(), contours, -1, (0,255,0), 2)
    draw_best = cv2.drawContours(draw_all, [best_contour], -1, (0,0,255), 5) if best_contour is not None else draw_all
    im_contour = ax_img4.imshow(preprocess.opencv2pil(draw_best))

    def update(val=None):
        cbs = checkbox.get_status()
        proc = preprocess.preprocess_image(
            orig_img, max_size=1000.0,
            padding=int(slider_padding.val),
            reduce_lighting_=cbs[0],
            gray=cbs[1],
            contrast=slider_contrast.val,
            exposure=slider_exposure.val,
            black_point_threshold=100 if cbs[2] else None,
            highlight_increase=100 if cbs[3] else None
        )
        im_proc.set_data(proc if len(proc.shape)==2 else cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
        im_proc.set_cmap('gray' if len(proc.shape)==2 else None)
        gray_img = proc if len(proc.shape)==2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray_img, int(slider_canny1.val), int(slider_canny2.val))
        if cbs[4]:
            k = np.ones((int(slider_closing.val), int(slider_closing.val)), np.uint8)
            closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(slider_gradient.val), int(slider_gradient.val)))
            grad = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, k2)
        else:
            grad = canny
        im_canny.set_data(grad)
        im_canny.set_cmap('gray')

        cnt, best_contour, contours, best_hull = contour.find_document_contour(resize_img, grad)
        draw_all = cv2.drawContours(resize_img.copy(), contours, -1, (0,255,0), 2)
        draw_best = cv2.drawContours(draw_all, [best_contour], -1, (0,0,255), 5) if best_contour is not None else draw_all
        im_contour.set_data(preprocess.opencv2pil(draw_best))
        fig.canvas.draw_idle()

    for slider in [slider_contrast, slider_exposure, slider_padding, slider_canny1, slider_canny2, slider_closing, slider_gradient]:
        slider.on_changed(update)
    checkbox.on_clicked(update)

    plt.show()
