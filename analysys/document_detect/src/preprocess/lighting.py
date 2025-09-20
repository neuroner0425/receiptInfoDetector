import cv2
import numpy as np

def reduce_lighting(img: np.ndarray, sigma: float = 10.0, gamma1: float = 0.3, gamma2: float = 1.5) -> np.ndarray:
    """
    조명(illumination)을 줄이는 homomorphic filter 적용.

    Args:
        img: OpenCV 이미지 (BGR)
        sigma: 가우시안 필터의 표준편차
        gamma1: 저주파 성분 계수
        gamma2: 고주파 성분 계수

    Returns:
        조명이 보정된 BGR 이미지
    """
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = img_YUV[:, :, 0]
    rows, cols = y.shape

    img_log = np.log1p(np.array(y, dtype='float') / 255)
    M, N = 2 * rows + 1, 2 * cols + 1

    X, Y = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    Xc, Yc = np.ceil(N / 2), np.ceil(M / 2)
    gaussian_numerator = (X - Xc) ** 2 + (Y - Yc) ** 2

    lpf = np.exp(-gaussian_numerator / (2 * sigma * sigma))
    hpf = 1 - lpf
    lpf_shift = np.fft.ifftshift(lpf)
    hpf_shift = np.fft.ifftshift(hpf)

    img_fft = np.fft.fft2(img_log, (M, N))
    img_lf = np.real(np.fft.ifft2(img_fft * lpf_shift, (M, N)))
    img_hf = np.real(np.fft.ifft2(img_fft * hpf_shift, (M, N)))

    img_adjust = gamma1 * img_lf[0:rows, 0:cols] + gamma2 * img_hf[0:rows, 0:cols]
    img_exp = np.expm1(img_adjust)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp) + 1e-6)
    img_out = np.array(255 * img_exp, dtype='uint8')

    img_YUV[:, :, 0] = img_out
    return cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
