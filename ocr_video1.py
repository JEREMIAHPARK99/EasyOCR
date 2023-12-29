
import os
import pandas as pd
from reader import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image, ImageOps, ImageFilter
from reader import Reader
from scipy.ndimage import gaussian_filter
#import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#한글 부분
#labeled_coordinates = {'상태': [619, 79, 257, 41], '시간자동': [970, 75, 144, 39]}
#소스에서 윗부분의 좌표
labeled_coordinates = {'날짜':[1341, 67, 127, 33], '시간':[1338, 102, 125, 33], '형체위치':[541, 150, 68, 26], 
    '에젝터위치':[547, 187, 71, 25], '금형두께':[555, 221, 67, 25], '형체력':[555, 253, 74, 28], '스크류위치':[788, 144, 79, 29], 
    '사출압력':[794, 183, 76, 26], '최대압력':[797, 218, 75, 27], '보압전환':[798, 253, 75, 29],'사출시간':[1075, 143, 82, 32],
    '계량시간':[1074, 182, 80, 30],'스크류속도':[1075, 222, 80, 27], 'InjCushion':[1072, 257, 80, 29],'전체시간':[1330, 148, 81, 30],
    '동작시간':[1328, 186, 77, 31], '생산수량':[1331, 227, 88, 26], '윤활1':[1398, 251, 38, 21], '윤활2':[1391, 272, 55, 19], 
    '윤활3':[1379, 289, 55, 18]}
# Define the custom folder to save cropped ROI images
#roi_save_folder = 'korean_crop'
roi_save_folder = 'crop_video1'
os.makedirs(roi_save_folder, exist_ok=True)

#인식에 사용할 이미지 소스
transformed_path = 'frames_folder'
#프레임 파일에서 첫 300개의 사진을 사용하기 위한 코드
image_files = sorted(os.listdir(transformed_path))
    #model_storage_directory=r'C:/Users/Hongjin/anaconda3/Lib/site-packages/easyocr/model/', #학습모델위치
    #user_network_directory=r'C:/Users/Hongjin/anaconda3/Lib/site-packages/easyocr/user_network/', #yaml 위치
    #recog_network='custom'#
#easyOCR 리더 설정
reader = Reader(['en'], gpu=True)
#reader = Reader(['ko'], gpu=True)

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

batch_size = len(image_files)

roi_runtime_df = pd.DataFrame(columns=list(labeled_coordinates.keys()))

# 시간을 재기 위한 함수 설정하는 코드
def save_elapsed_time(elapsed_time, file_path):
    with open(file_path, 'w') as file:
        file.write(f'Elapsed time: {elapsed_time:.2f} seconds')
"""
#출력되는 결과 정리하는 코드
def clean_ocr_result(ocr_text):
    # 숫자가 아닌 문자, 빈 space, 특수 문자 제외
    ocr_text = ''.join(char for char in ocr_text if char.isnumeric() or char == '.')
    # 여러 개의 점을 하나의 .로 대체
    ocr_text = re.sub(r'\.{2,}', '.', ocr_text)
    # 문자 앞의 점이나 뒤에 오는 점들 모두 제거하도록 설정
    ocr_text = ocr_text.strip('. ')
    # 여러 개로 나누어져 있는 문자를 하나로 정리하고 앞에 비어있는 문자도 정리
    ocr_text = ' '.join(ocr_text.split()).strip()
    # 결과가 비어있거나 .만 포함하면 결과를 0으로 후처리하기
    if not ocr_text or all(char == ' ' for char in ocr_text):
        ocr_text = '0'
    return ocr_text
"""

# OCR 결과와 신뢰 결과가 출력되는 것을 구분하여 출력하는 함수
def process_image(image_file, labeled_coordinates, image_number):
    ocr_texts, confidence_scores = process_roi_batch(image_file, labeled_coordinates, list(labeled_coordinates.items()), 
                                                     image_number)
    runtime_texts, _ = process_roi_batch(image_file, labeled_coordinates, list(labeled_coordinates.items()), 
                                            image_number)
    # 데이터프레임에 OCR 결과를 포함하는 새로운 행을 추가하는 부분
    result_df.loc[image_number] = ocr_texts
    # 데이터프레임에 신뢰를 포함하는 새로운 행을 추가하는 부분
    confidence_df.loc[image_number] = confidence_scores
    # OCR 런타임을 데이터프레임에 추가하는 부분
    roi_runtime_df.loc[image_number] = runtime_texts
        
# 각 ROI에서 결과를 받아들이고 OCR 텍스트 인식 결과를 저장하도록 처리하는 함수
def process_roi_batch(image_file, labeled_coordinates, coordinates_list, image_number):
    image_path = os.path.join(transformed_path, image_file)
    img = Image.open(image_path)
    ocr_results = {row_header: [] for row_header in labeled_coordinates.keys()}
    runtime_results = {row_header: [] for row_header in labeled_coordinates.keys()}
    ocr_texts = [] #OCR 결과를 저장할 리스트
    confidence_scores = [] #신뢰 점수를 저장할 리스트 설정
    for row_header, coordinates in coordinates_list:
        x, y, width, height = coordinates
        roi = img.crop((x, y, x + width, y + height))
        roi_gray = np.array(roi.convert('L'))
        roi_blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        # ROI 영역별 이미지 결과들을 저장할 경로
        save_path = os.path.join(roi_save_folder, f'{image_file}_{image_number}_{row_header}.jpg')
        # 각 ROI영역별 이미지 저장
        roi.save(save_path)
        #문자 추출하는데 걸리는 시간 측정하기 위해 시간 측정 시작
        ocr_start_time = time.time()
        result = reader.readtext(roi_blurred, detail=1)
        end_time = time.time() - ocr_start_time
        # OCR결과와 신뢰 점수를 따로 출력하는 코드
        if result:
            ocr_text = result[0][1]  # OCR 결과
            confidence = result[0][2]  # 신뢰 점수
        else:
            ocr_text = '0'
            confidence = 0.0
        # 반점등을 포함한 기호들을 모두 .으로 대체하기
        ocr_text = re.sub(r'[^a-zA-Z0-9. ]', '', ocr_text).replace(',', '.')
        #ocr_texts에 ocr 결과 추가하고 confidence_scores에 신뢰 점수 추가
        ocr_texts.append(ocr_text)
        confidence_scores.append(confidence)
        runtime_results[row_header].append(end_time)
    # Calculate the mean runtime for each ROI and return the mean runtime dictionary
    mean_runtime_dict = {row_header: np.mean(runtimes) for row_header, runtimes in runtime_results.items()}
    runtime_df.loc[image_number] = list(mean_runtime_dict.values())  # Save the mean runtimes to the DataFrame
    
    return ocr_texts, confidence_scores

# 이미지 처리하고 소스에서 문자 추출하는 코드 부분
def process_labeled_coordinates(labeled_coordinates, image_files):
    ocr_results = {row_header: [] for row_header in labeled_coordinates.keys()}
    for i in range(0, len(image_files), batch_size):
        start = i
        end = min(i + batch_size, len(image_files))
        # 모든 ROI를 병렬 처리하여 인식하는 부분
        coordinates_list = list(labeled_coordinates.items())[start:end]
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_roi_batch, image_file, labeled_coordinates, coordinates_list)
                for image_file in image_files[start:end]
            ]
            for future in futures:
                result = future.result()
                for row_header, ocr_text in result.items():
                    ocr_results.get(row_header).extend(ocr_text)
    return ocr_results
            
# ROI되는 영역의 이름을 리스트에 저장하는 코드
roi_names = list(labeled_coordinates.keys())
# 데이터프레임 생성 전에 빈 데이터프레임 생성
runtime_df = pd.DataFrame(columns=list(labeled_coordinates.keys()))

# Create an empty DataFrame with the desired structure
result_df = pd.DataFrame(columns=list(labeled_coordinates.keys()))
confidence_df = pd.DataFrame(columns=roi_names)
    
#OCR 과정의 소요시간 측정
start_time = time.time()

# 이미지를 처리하고 데이터프레임 생성
for i, image_file in enumerate(image_files):
    process_image(image_file, labeled_coordinates, i + 1)
    
# OCR 전체 소요시간 측정
elapsed_time_up_to_ocr = time.time() - start_time

# 데이터프레임에 ocr결과 출력
print("OCR Results DataFrame:")
print(result_df)

# 신뢰 점수를 표현한 데이터프레임
print("Confidence Scores DataFrame:")
print(confidence_df)

print("Runtime dataframe: ")
print(runtime_df)

# 위 데이터프레임을 csv 파일로 저장
ocr_result_xls_path = 'video1_ocr.xlsx' #뒤에 기호 인식 안한 코드
confidence_scores_xls_path = 'video1_confidence.xlsx' #뒤에 기호 인식 안한 코드
roi_runtimes_xls_path = 'video1_runtimes.xlsx'
result_df.to_csv(ocr_result_xls_path, index_label='Image Number')
confidence_df.to_csv(confidence_scores_xls_path, index_label='Image Number')
roi_runtime_df.to_csv(roi_runtimes_xls_path, index_label='Image Number')

print(f'total time taken for OCR: {elapsed_time_up_to_ocr:.4f} seconds')

# ocr 수행시간을 저장하는 코드
#text_file_path = 'elapsed_time.txt'
#save_elapsed_time(elapsed_time_up_to_ocr, text_file_path)
#print(f'Elapsed time written to: {text_file_path}')