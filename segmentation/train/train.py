
# 경로 설정
images_dir = "output/preprocess"  # Grayscale 이미지 경로
masks_dir = "output/mask_data"  # 마스크 이미지 경로

# 데이터셋 및 DataLoader 생성
dataset = WrinkleSegmentationDataset(images_dir, masks_dir, target_size=(512, 512))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)


for images, masks in dataloader:
    print("Images batch shape:", images.shape)
    print("Masks batch shape:", masks.shape)
    break  # 첫 배치만 확인




# 모델 초기화
model = UNet(in_channels=1, out_channels=2).to("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 손실 및 최적화 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")



# 모델 저장
torch.save(model.state_dict(), "unet_model_epoch_100.pth")
print("Model training complete and saved.")
