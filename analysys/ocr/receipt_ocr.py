from paddleocr import PaddleOCR
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math

# filenames = [f for f in os.listdir('./receipt_data/train/img')
#              if os.path.isfile(os.path.join('./receipt_data/train/img', f)) and
#              f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]
# filenames = ['4.jpg', '6.jpg','10.png', '13.jpg', '15.jpg']
filenames = ['16.jpg']
print(filenames)

# ----------------------------------------------

if not os.path.exists('.official_models'):
    os.makedirs('.official_models')
    # Download the PaddleOCR model if it doesn't exist
    
ocr = PaddleOCR(
    lang='korean',
    # 문서 방향 분류 모델 (문서가 회전되어 있을 때 방향을 판별)
    doc_orientation_classify_model_dir='.official_models/PP-LCNet_x1_0_doc_ori',
    # 문서 왜곡 보정(펴기) 모델 (구겨지거나 휘어진 문서를 펴줌)
    doc_unwarping_model_dir='.official_models/UVDoc',
    # 텍스트 영역 검출 모델 (이미지에서 글자가 있는 영역을 찾음)
    # text_detection_model_dir='.official_models/PP-OCRv5_server_det',
    # 텍스트 라인 방향 분류 모델 (텍스트 줄의 방향을 판별)
    textline_orientation_model_dir='.official_models/PP-LCNet_x1_0_textline_ori',
    # 텍스트 인식(문자 추출) 모델 (검출된 영역에서 실제 글자를 읽음)
    # text_recognition_model_dir='.official_models/PP-OCRv5_server_rec',
)

# result = ocr.predict_iter(
#     filenames,  # 처리할 이미지 파일 리스트
#     use_doc_orientation_classify=None,  # 문서 방향 분류 사용 여부 (None이면 자동)
#     use_doc_unwarping=None,             # 문서 왜곡 보정 사용 여부 (None이면 자동)
#     use_textline_orientation=None,      # 텍스트 라인 방향 분류 사용 여부 (None이면 자동)
#     text_det_limit_side_len=None,       # 텍스트 검출 시 입력 이미지의 최대 한 변 길이
#     text_det_limit_type=None,           # 텍스트 검출 시 한 변 제한 방식
#     text_det_thresh=None,               # 텍스트 검출 임계값
#     text_det_box_thresh=None,           # 텍스트 박스 임계값
#     text_det_unclip_ratio=None,         # 텍스트 박스 확장 비율
#     text_rec_score_thresh=None,         # 텍스트 인식 결과 임계값
# )

def preprocess_image(filename):
    file_name, _ = os.path.splitext(filename)
    img_path = os.path.join('receipt_data', 'img', filename)
    result_path = os.path.join(
        'receipt_data', 'ocr_result', file_name, 'preprocessed.jpg')
    
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")

    # Resize image if any side is greater than 4000 pixels
    max_side = max(image.shape[:2])
    max_size = 1200
    if max_side > max_size:
        print(f"Resizing image {filename} from {max_side} to {max_size} pixels.")
        scale = max_size / max_side
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path, image)

    # if False: # Change to True to skip preprocessing
        # return img_path
    return result_path

def doOCR(ocr, filename):
    print(f"Processing {filename}...")
    
    file_name, _ = os.path.splitext(filename)
    result_path = os.path.join(
        'receipt_data', 'ocr_result', file_name)

    img_path = preprocess_image(filename)
    
    if not os.path.exists(img_path):
        print(f"Image file {img_path} does not exist.")
        return
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_iter = ocr.predict_iter(
        img_path,   # 처리할 이미지 파일
        use_doc_orientation_classify=True,  # 문서 방향 분류 사용 여부 (None이면 자동)
        use_doc_unwarping=True,             # 문서 왜곡 보정 사용 여부 (None이면 자동)
        use_textline_orientation=None,      # 텍스트 라인 방향 분류 사용 여부 (None이면 자동)
        text_det_limit_side_len=None,       # 텍스트 검출 시 입력 이미지의 최대 한 변 길이
        text_det_limit_type=None,           # 텍스트 검출 시 한 변 제한 방식
        text_det_thresh=0,               # 텍스트 검출 임계값
        text_det_box_thresh=None,           # 텍스트 박스 임계값
        text_det_unclip_ratio=None,         # 텍스트 박스 확장 비율
        text_rec_score_thresh=None,         # 텍스트 인식 결과 임계값
    )
    
    for result in result_iter:
        # print(help(result))
        # result.save_all(result_path)
        result.save_to_img(result_path)
        result.save_to_json(result_path)
        # json_result = result.json
        # print(result.str)
        # print(result.items())
    
    print(f"Results saved to {result_path}")


for filename in filenames:
    doOCR(ocr, filename)
