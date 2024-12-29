import numpy as np
import open3d as o3d

def create_segmentation_array(x, y):
    seg_array = []
    for i in range(x[0], x[1] + 1):
        for j in range(y[0], y[1] + 1):
            seg_array.append([i, j])
    return seg_array

def calculate_surface_area(pcd_true, x, y):
    seg_array = create_segmentation_array(x, y)

    pcd_segmented = []
    for i in range(len(seg_array)):
        [seg_x, seg_y] = seg_array[i]
        pcd_segmented.append(pcd_true[seg_y * 192 + 192 - seg_x])

    pcd_segmented = np.array(pcd_segmented, dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_segmented)

    # 노멀(법선 벡터) 계산
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    # Poisson Surface Reconstruction 메쉬 생성
    mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    area = mesh1.get_surface_area()
    return area

# 사용 예시
if __name__ == "__main__":
    # pcd_true와 x, y 값을 정의해야 합니다.
    pcd_true = np.random.rand(192 * 192, 3)  # 예시 데이터
    x = [74, 113]
    y = [114, 165]
    
    area = calculate_surface_area(pcd_true, x, y)
    print("Surface area of mesh1:", area)