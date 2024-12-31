# README

## 3D 표면적 계산 API

이 프로젝트는 입력 이미지와 대응되는 깊이 데이터를 기반으로 분할 영역의 3D 표면적을 계산하는 FastAPI 기반 애플리케이션입니다. Open3D를 사용하여 3D 포인트 클라우드 및 메쉬 처리를 수행하며, 이미지 분할에는 사전 학습된 UNet 모델을 사용합니다.

---

## 주요 기능

- **깊이 데이터를 포인트 클라우드로 변환**: 깊이 데이터를 3D 포인트 클라우드로 변환합니다.
- **UNet 기반 이미지 분할**: 입력 이미지에서 관심 영역을 분할합니다.
- **3D 표면적 계산**: 분할된 영역의 3D 메쉬를 기반으로 표면적을 계산합니다.
- **REST API 제공**: HTTP POST 요청을 통해 전체 프로세스를 수행합니다.

---

## 요구 사항

### 소프트웨어 및 라이브러리

- **Python**: >=3.10
- **FastAPI**: 웹 프레임워크
- **Open3D**: 3D 포인트 클라우드 및 메쉬 처리
- **PyTorch**: 딥러닝 프레임워크
- **NumPy**: 수치 계산
- **Matplotlib**: 시각화
- **Pillow**: 이미지 처리
- **cv2**: OpenCV를 활용한 이미지 전처리

### 하드웨어

- **GPU (선택 사항)**: PyTorch(CUDA)를 활용한 빠른 분할 처리

---

## 설치 방법

1. **저장소 클론**:

   ```bash
   git clone https://github.com/your-repo/3d-surface-area-api.git
   cd 3d-surface-area-api
   ```

2. **가상 환경 설정**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows에서는 venv\Scripts\activate
   ```

3. **의존성 설치**:

   ```bash
   pip install -r requirements.txt
   ```

4. **사전 학습된 모델 준비**:
   사전 학습된 UNet 모델을 `segmentation/models/unet_model_epoch_100.pth` 경로에 배치합니다.

5. **애플리케이션 실행**:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

---

## API 엔드포인트

### POST `/api/v1.0/calculate_area`

분할 영역의 3D 표면적을 계산합니다.

#### 요청

- **image\_file**: 분할을 위한 그레이스케일 이미지 파일 (PNG 형식)
- **depth\_file**: 깊이 데이터를 포함하는 JSON 파일

#### cURL 요청 예시

```bash
curl -X POST "http://localhost:8080/api/v1.0/calculate_area" \
-F "image_file=@data/wrinkle_test_4.png" \
-F "depth_file=@data/depth_data.json"
```

#### 응답

- **`area_3d`**: 3D에서 분할된 영역의 계산된 표면적

#### 응답 예시

```json
{
  "area_3d": 123.45
}
```

---

## 파일 구조

```
project-root/
├── depth/
│   ├── depth_area.py       # 깊이 데이터 처리 로직
├── segmentation/
│   ├── models/             # 사전 학습된 UNet 모델
│   ├── inference/          # 분할 추론 로직
├── main.py                 # FastAPI 애플리케이션 엔트리 포인트
├── requirements.txt        # Python 의존성 목록
├── README.md               # 문서화 파일
└── data/                   # 예제 데이터
```

---

## 작동 방식

1. **입력**:

   - 분할을 위한 그레이스케일 이미지
   - JSON 형식의 깊이 데이터

2. **이미지 분할**:

   - UNet 모델을 사용하여 관심 영역을 분할합니다.

3. **깊이 데이터 처리**:

   - 깊이 JSON 데이터를 3D 포인트 클라우드로 변환합니다.

4. **면적 계산**:

   - 분할된 영역을 포인트 클라우드에 매핑합니다.
   - 3D 메쉬를 생성하고 표면적을 계산합니다.

---

## 개발 참고 사항

- **테스트**:
  Postman이나 cURL을 사용하여 제공된 `data/` 디렉토리의 예제 데이터를 활용해 API를 테스트하세요.

- **에러 처리**:
  입력 파일의 유효성을 검증하고 누락되거나 잘못된 데이터를 처리할 수 있도록 구현하세요.

- **최적화**:

  - GPU를 활용하면 분할 속도가 향상됩니다.
  - 로컬 캐싱 전략을 조사하여 지연 시간을 줄이세요.

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.
