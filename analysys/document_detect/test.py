import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import detect, preprocess, reshape, util

DATASET_PATH = './dataset'

def detect_documents(dataset: str):
    input_dir = os.path.join(DATASET_PATH, dataset)
    if not os.path.isdir(input_dir):
        print(f"Not a directory: {input_dir}")
        return

    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

    if not img_files:
        print(f"No images found in {input_dir}")
        return

    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"Processing {img_path}...")
        origin_img = cv2.imread(img_path)
        if origin_img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        resize_img = preprocess.resize_image(origin_img.copy(), 1000.0)
        
        real_time_param = False
        
        if real_time_param:
            detect.realtime_edge_dect(origin_img)

        else:
            save_steps_dir = os.path.join(input_dir, "steps")
            os.makedirs(save_steps_dir, exist_ok=True)

            # 실제로 저장하기
            preprocessed = preprocess.preprocess_image(
                origin_img,
                max_size=1000.0,
                padding=100,
                reduce_lighting_=True,
                gray=False,
                contrast=1.50,
                exposure=-40,
                black_point_threshold=None,
                highlight_increase=None,
                show_steps=False,
                save_steps=save_steps_dir
            )

            candy = cv2.Canny(preprocessed, 125, 150)
            cv2.imwrite(f'{save_steps_dir}/candy.jpg', candy)

            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(candy, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(f'{save_steps_dir}/closing.jpg', closing)

            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edge_img = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel2)
            cv2.imwrite(f'{save_steps_dir}/edge.jpg', edge_img)

            # 윤곽선 검출
            screen_cnt, wild_contour, contours, s_contour = detect.find_document_contour(
                resize_img, 
                edge_img, 
                save_dir=save_steps_dir, 
                show_all=False
            )
            draw_all = cv2.drawContours(resize_img.copy(), contours, -1, (0,255,0), 2)
            cv2.imwrite(f'{save_steps_dir}/all_contours.jpg', draw_all)
            
            draw_best_contour = cv2.drawContours(resize_img.copy(), [s_contour], -1, (0,0,255), 2) if wild_contour is not None else resize_img
            cv2.imwrite(f'{save_steps_dir}/best_contours.jpg', draw_best_contour)
            cv2.imwrite(f'out/{dataset}.jpg', draw_best_contour)
            
            if screen_cnt is not None and s_contour is not None:
                rectified = reshape.rectify_document(
                    original_image=origin_img,
                    screen_cnt=screen_cnt,
                    s_contour=s_contour,
                    contour_shape=edge_img.shape[:2]
                )

                cv2.imwrite(f'{save_steps_dir}/reshape.jpg', rectified)
                cv2.imwrite(f'out/{dataset}_reshape.jpg', rectified)
                cv2.imshow("Final Unwarped Document", rectified)
                cv2.waitKey(0); cv2.destroyAllWindows()
                
            else:
                print("문서 컨투어 탐지 실패")

if __name__ == "__main__":
    datasets = os.listdir(DATASET_PATH)
    print("Available datasets:", datasets)
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        detect_documents(dataset)
