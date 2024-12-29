import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2  
import matplotlib.pyplot as plt
from train.model import UNet 


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
    model_path = 'models/unet_model_epoch_100.pth'
    model = UNet(in_channels=1, out_channels=2).to(device)  # UNet 모델 초기화
    model.load_state_dict(torch.load(model_path, map_location=device))  # 가중치 로드
    model.eval()  # 평가 모드`로 설정
    return model



def segmented_area(image_path, model_path):
    # 사용 예시
    image_path = 'data/wrinkle_test_4.png'  # 테스트할 이미지 경로
    model_path = "models/unet_model_epoch_100.pth"
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

        for i in range(len(images)):
            plt.figure(figsize=(12, 4))
            
            # 입력 이미지
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(images[i].cpu().squeeze(), cmap="gray")
            plt.axis("off")
            
            # Predicted Mask Overlay
            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask Overlay")
            
            # 기본 이미지와 예측 마스크를 덧대기
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
            
            

def visualize_and_save_results(images, predicted_masks, output_dir):
    for i in range(len(images)):
        plt.figure(figsize=(12, 4))
        
        # 입력 이미지
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(images[i].cpu().squeeze(), cmap="gray")
        plt.axis("off")
        
        # Predicted Mask Overlay
        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask Overlay")
        
        # 기본 이미지와 예측 마스크를 덧대기
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
            
def convert_to_depth_coordinates(seg_x, seg_y, seg_image_size=(512, 512), depth_map_size=(256, 192)):
    seg_width, seg_height = seg_image_size
    depth_width, depth_height = depth_map_size

    width_ratio = 1920 / seg_width
    height_ratio = 1440 / seg_height

    depth_x = int(seg_x * (depth_width / seg_width))
    depth_y = int(seg_y * (depth_height / seg_height))

    return depth_x, depth_y

# 세그멘테이션된 좌표 예시
seg_array = [[100, 200], [150, 250]]  # 예시 좌표
depth_coordinates = []

for seg_x, seg_y in seg_array:
    depth_x, depth_y = convert_to_depth_coordinates(seg_x, seg_y)
    depth_coordinates.append((depth_x, depth_y))

print(depth_coordinates)  # 변환된 depth map 좌표 출력