from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from depth.depth_area import calculate_surface_area
from segmentation.inference.utils import segmented_area
from .visualize_area import visualize_area

import os
import cv2

app = FastAPI()


@app.post("/api/v1.0/calculate_area")
async def calculate_area(
    image_file: UploadFile = File(...), depth_file: UploadFile = File(...)
):
    # 이미지 파일을 임시로 저장
    image_path = f"temp/{image_file.filename}"
    os.makedirs("temp", exist_ok=True)  # temp 디렉토리 생성
    with open(image_path, "wb") as f:
        f.write(await image_file.read())

    # 포인트 클라우드 JSON 파일을 임시로 저장
    depth_path = f"temp/{depth_file.filename}"
    with open(depth_path, "wb") as f:
        f.write(await depth_file.read())

    # segmented_area 함수 호출
    depth_coordinates = segmented_area(image_path)
    print("depth_coordinates is : ", len(depth_coordinates))

    area_3d = calculate_surface_area(depth_coordinates, depth_path)
    area_3d = area_3d * 10000
    # 결과 이미지에 area_3d 표시
    result_image_path = "output/result.png"

    visualize_area(area_3d, result_image_path)

    return FileResponse(result_image_path, media_type="image/png")
