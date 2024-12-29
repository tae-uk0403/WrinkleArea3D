import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class WrinkleSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size

        # 이미지 및 마스크 파일 정렬
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))
        
        # 이미지와 마스크 개수 일치 여부 확인
        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("이미지와 마스크의 파일 수가 일치하지 않습니다.")

        # 이미지와 마스크를 같은 크기로 변환
        self.resize_transform = transforms.Resize(self.target_size)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 이미지와 마스크 경로 설정
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        # Grayscale 이미지 및 마스크 불러오기 및 크기 변환
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # 이미지와 마스크 크기 조정
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)

        # 이미지를 Numpy 배열로 변환 후 uint8 형식으로 강제 설정
        image = np.array(image)
        mask = np.array(mask)

        # Tensor 변환

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # 0-1 범위로 정규화
        mask = torch.from_numpy(mask).long()  # 마스크는 long 타입으로 설정

        return image, mask
