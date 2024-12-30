import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2  
import matplotlib.pyplot as plt
import sys

from ..train.model import UNet 


def preprocess_image_for_test(image_path, target_size=(512, 512)):
    """
    inference를 위한 전처리 및 tensor 변환 함수
    """
    
    # 이미지 불러오기 및 Grayscale로 변환
    image = Image.open(image_path).convert("L")

    # 이미지 크기 조정
    resize_transform = transforms.Resize(target_size)
    image = resize_transform(image)

    # 이미지를 Numpy 배열로 변환 후 uint8 형식으로 강제 설정
    image = np.array(image)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(32, 32))
    clahe_image = clahe.apply(image)

    # CLAHE 적용된 이미지를 Tensor 변환 및 정규화
    image_tensor = torch.from_numpy(clahe_image).float().unsqueeze(0) / 255.0  # 0-1 범위로 정규화

    # 배치 차원 추가
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가

    return image_tensor


def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=2).to(device)  # UNet 모델 초기화
    model.load_state_dict(torch.load(model_path, map_location=device))  # 가중치 로드
    model.eval()  # 평가 모드`로 설정
    return model



def segmented_area(image_path):
    # 사용 예시
    image_path = 'segmentation/data/wrinkle_test_image.jpg'  # 테스트할 이미지 경로
    model_path = "segmentation/models/unet_model_epoch_100.pth"
    processed_image = preprocess_image_for_test(image_path)
    print("Processed image shape:", processed_image.shape)  # torch.Size([1, 1, 512, 512])


    with torch.no_grad():
        images = processed_image
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        images = images.to(device)
        
        # inference
        model = load_model(model_path, device)
        outputs = model(images)
        predicted_masks = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 결과 저장
        output_dir = 'output'  # 결과를 저장할 경로
        os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성
        
    visualize_and_save_results(images, predicted_masks, output_dir)
        
    print("predicted_masks is : ", predicted_masks.shape)
    depth_coordinates = convert_to_depth_coordinates(predicted_masks, 1, seg_image_size=(512, 512), depth_map_size=(256, 192))
    visualize_depth_coordinates(depth_coordinates) 
    return depth_coordinates
    
    
def visualize_and_save_results(images, predicted_masks, output_dir):
    for i in range(len(images)):
        plt.figure(figsize=(6, 6))  # 크기 조정 (필요에 따라 변경 가능)
        
        # Predicted Mask Overlay
        plt.title("Predicted Mask Overlay")
        
        # 예측 마스크를 컬러로 변환 (예: 빨간색)
        input_image = images[i].cpu().squeeze().numpy()  # 입력 이미지를 Numpy 배열로 변환
        predicted_mask = predicted_masks[i]  # 예측 마스크
        
        # 예측 마스크를 컬러로 변환 (예: 빨간색)
        overlay = np.zeros_like(input_image)
        overlay[predicted_mask > 0] = 1  # 예측된 마스크가 1인 부분을 강조
        
        # 입력 이미지에 오버레이 추가
        plt.imshow(input_image, cmap="gray")
        plt.imshow(overlay, alpha=0.5, cmap="Reds")  # 빨간색으로 오버레이
        
        plt.axis("off")
        
        # 결과 저장
        output_path = os.path.join(output_dir, f'result_{i}.png')  # 결과 파일 경로
        plt.savefig(output_path, bbox_inches='tight')  # 결과 저장
        plt.close()  # 현재 플롯 닫기
            
            
def convert_to_depth_coordinates(predicted_masks, segmentation_class, seg_image_size=(512, 512), depth_map_size=(256, 192)):
    # predicted_masks에서 특정 클래스의 픽셀 좌표 찾기
    seg_coords = np.argwhere(predicted_masks[0] == segmentation_class)  # 클래스의 픽셀 좌표

    depth_coordinates = []
    seg_width, seg_height = seg_image_size
    depth_width, depth_height = depth_map_size

    for coord in seg_coords:
        y, x = coord  # (y, x) 형식으로 좌표 추출
        depth_x = int(x * (depth_width / seg_width))
        depth_y = int(y * (depth_height / seg_height))
        depth_coordinates.append((depth_x, depth_y))

    return depth_coordinates


def visualize_depth_coordinates(depth_coordinates, seg_image_size=(512, 512)):
    # 깊이 좌표를 시각화하는 함수
    depth_x, depth_y = zip(*depth_coordinates)  # x와 y 좌표 분리

    plt.figure(figsize=(8, 8))
    plt.scatter(depth_x, depth_y, c='red', marker='o', s=10)  # 깊이 좌표를 빨간 점으로 표시    
    plt.xlim(0, seg_image_size[0])  # x축 범위 설정
    plt.ylim(0, seg_image_size[1])  # y축 범위 설정
    plt.title("Depth Coordinates Visualization")
    plt.xlabel("Depth X")
    plt.ylabel("Depth Y")
    plt.gca().set_aspect('equal', adjustable='box')  # 비율 유지
    plt.grid(True)
    plt.savefig("visualize_image_resize_to_depth.png", bbox_inches='tight')  # 결과 저장
    plt.close()  # 현재 플롯 닫기
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    image_path = 'data/wrinkle_test_4.png'
    model_path = 'models/unet_model_epoch_100.pth'
    segmented_area(image_path, model_path)

