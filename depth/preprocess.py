import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
import json

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
    
    # 포인트 클라우드 데이터를 Open3D 포인트 클라우드 형식으로 변환
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd_true)
    
    return pcd_true, o3d_pcd