import numpy as np
import open3d as o3d
import json
from segmentation.inference.utils import segmented_area

def depth_to_ply(pointcloud_path):
    pcd = []
    
    with open(pointcloud_path, "r") as f:
        depth_json = json.load(f)
    
        depth_w = 256
        depth_h = 192
        fl_x = depth_json["fl"]["x"]
        fl_y = depth_json["fl"]["y"]
        rgb_depth_ratio = 7.5
    
        camera_depth_array = np.zeros((depth_w, depth_h), dtype=np.float64)
        for i in range(len(depth_json['Depth'])):
            x = i % depth_w
            y = i // depth_w
            camera_depth_array[x][y] = depth_json['Depth'][i]
    
        # camera_xyz_array 생성
        for i in range(depth_w):
            for j in range(depth_h):
                z = camera_depth_array[i][j]
                x = -(i - depth_w / 2) * z / (fl_x / rgb_depth_ratio)
                y = -(j - depth_h / 2) * z / (fl_y / rgb_depth_ratio)
                pcd.append([x, y, z])
    
    pcd_true = np.array(pcd, dtype=np.float32)
    
    # 포인트 클라우드 데이터를 파일로 저장
    output_path = "output.ply"  # 저장할 파일 경로
    o3d.io.write_point_cloud(output_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_true)))
    
    return pcd_true



def create_seg_array(depth_coordinates):
    # depth_coordinates를 create_segmentation_array와 유사한 형태로 변환
    seg_array = [[coord[0], coord[1]] for coord in depth_coordinates]  # 각 좌표를 직접 추가
    return seg_array  # 변환된 segmentation_array 반환


def calculate_surface_area(depth_coordinates, pointcloud_path):
    seg_array = create_seg_array(depth_coordinates)
    pcd_true = depth_to_ply(pointcloud_path)
    
    
    pcd_segmented = []

    for i in range(len(seg_array)):
        [seg_x, seg_y] = seg_array[i]
        pcd_segmented.append(pcd_true[seg_y * 192 + 192 - seg_x])
    
    pcd_segmented = np.array(pcd_segmented, dtype=np.float32)
    print("pcd_segmented is ", pcd_segmented)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_segmented)
    # 노멀(법선 벡터) 계산
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    # Poisson Surface Reconstruction 메쉬 생성
    mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    area = mesh1.get_surface_area()
    print("area is ", area)
    return area

