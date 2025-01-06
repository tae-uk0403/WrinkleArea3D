import cv2


def visualize_area(area_3d, result_image_path):
    image = cv2.imread(result_image_path)

    # 텍스트 추가
    text = f"3D wrinkle area = {area_3d:.2f}"  # 소수점 2자리까지 표시

    # 텍스트 크기 계산 (1.5배 증가)
    font_scale = 3  # 폰트 크기 1.5배
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    box_coords = ((10, 1850), (10 + text_width + 10, 1850 - text_height - 20))

    # 텍스트 박스 그리기
    cv2.rectangle(
        image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED
    )  # 박스 색상: 검정색
    cv2.putText(
        image,
        text,
        (10, 1840),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,  # 폰트 크기 1.5배
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )

    cv2.imwrite(result_image_path, image)
