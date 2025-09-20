import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def realtime_preprocess_image(
    image: np.ndarray,
    initial_max_size: float = 1000.0,
    initial_padding: int = 0,
):
    """
    Matplotlib 슬라이더/체크박스로 실시간 전처리 파라미터 조절 인터랙티브 UI를 실행합니다.

    Args:
        image (np.ndarray): 원본 BGR 이미지.
        initial_max_size (float): 리사이즈 최대 길이.
        initial_padding (int): 기본 패딩(여백) 픽셀.

    Returns:
        None (실시간 인터랙티브 UI 실행)
    """
    
    from ..preprocess import preprocess_image

    orig_img = image.copy()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.38)
    
    # 슬라이더/체크박스 생성
    slider_contrast = Slider(plt.axes([0.35, 0.28, 0.55, 0.03]), 'Contrast', 0.5, 3.0, valinit=1.0)
    slider_exposure = Slider(plt.axes([0.35, 0.23, 0.55, 0.03]), 'Exposure', -150, 150, valinit=0.0)
    slider_padding  = Slider(plt.axes([0.35, 0.18, 0.55, 0.03]), 'Padding', 0, 100, valinit=initial_padding, valstep=1)
    checkbox = CheckButtons(
        plt.axes([0.025, 0.6, 0.2, 0.28]),
        ['Reduce Lighting', 'Gray', 'Black Point', 'Highlight'],
        [False, False, False, False]
    )

    # 초기 전처리 결과 표시
    def get_processed():
        return preprocess_image(
            orig_img,
            max_size=initial_max_size,
            padding=slider_padding.val,
            reduce_lighting_=checkbox.get_status()[0],
            gray=checkbox.get_status()[1],
            contrast=slider_contrast.val,
            exposure=slider_exposure.val,
            black_point_threshold=100 if checkbox.get_status()[2] else None,
            highlight_increase=100 if checkbox.get_status()[3] else None,
            show_steps=False, save_steps=None
        )

    proc_img = get_processed()
    imshow_kwargs = {'cmap': 'gray'} if len(proc_img.shape) == 2 else {}
    im = ax.imshow(
        proc_img if len(proc_img.shape) == 2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB),
        **imshow_kwargs
    )
    ax.set_title('실시간 전처리 결과')
    ax.axis('off')

    # 갱신 이벤트 함수
    def update(val=None):
        out = get_processed()
        if len(out.shape) == 2:
            im.set_data(out)
            im.set_cmap('gray')
        else:
            im.set_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            im.set_cmap(None)
        fig.canvas.draw_idle()

    # 이벤트 연결
    slider_contrast.on_changed(update)
    slider_exposure.on_changed(update)
    slider_padding.on_changed(update)
    checkbox.on_clicked(update)

    plt.show()
