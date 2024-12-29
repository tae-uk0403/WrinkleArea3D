import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw


def preprocess_images(input_folder, output_folder):
    """
    Processes all images in the specified input folder by applying grayscale conversion and CLAHE,
    and saves them to the output folder.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save processed images.

    Returns:
        None
    """
    # Get a list of all files in the input folder
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # Save with the same filename

        # Step 1: Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: The image at path {image_path} could not be loaded.")
            continue

        # Step 2: Convert to Grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply CLAHE for contrast enhancement

        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(32, 32))
        clahe_image = clahe.apply(gray_image)
        # Step 4: Save the processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, clahe_image)
        print(f"Processed and saved: {output_path}")


def create_segmentation_masks_from_folder(json_folder, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더의 모든 JSON 파일 처리
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)

            # JSON 파일 읽기
            with open(json_path, 'r') as file:
                data = json.load(file)

            # 이미지 크기 정보 가져오기
            image_height = data['imageHeight']
            image_width = data['imageWidth']

            # 빈 마스크 생성 (0으로 초기화된 PIL 이미지)
            mask_image = Image.new("L", (image_width, image_height), 0)
            draw = ImageDraw.Draw(mask_image)

            # 각 주름 영역에 대해 Polygon 그리기
            for shape in data['shapes']:
                if shape['label'] == 'wrinkle':
                    # Polygon 좌표 가져오기
                    points = [(point[0], point[1]) for point in shape['points']]

                    # 마스크 이미지에 Polygon 그리기 (255로 채우기)
                    draw.polygon(points, outline=1, fill=1)

            # 마스크 이미지 저장
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            mask_image.save(output_path)
            print(f"Segmentation 마스크가 {output_path}에 저장되었습니다.")


# 사용 예시
json_folder = 'wrinkle_preprocess'  # JSON 파일이 있는 폴더 경로
output_folder = 'output/training_data'  # 마스크 이미지 저장할 폴더 경로
create_segmentation_masks_from_folder(json_folder, output_folder)
