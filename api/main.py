import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
import json




x=[74,113]
y=[114,165]
seg_array = []
for i in range(x[0],x[1]+1):
    for j in range(y[0],y[1]+1):
        seg_array.append([i,j])

print(seg_array)

pcd_segmented = []
for i in range(len(seg_array)):
    [seg_x, seg_y] = seg_array[i]
    pcd_segmented.append(pcd_true[seg_y *192 + 192 - seg_x])

pcd_segmented = np.array(pcd_segmented, dtype=np.float32)
print(len(pcd_segmented))
print(pcd_segmented)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_segmented)



# 노멀(법선 벡터) 계산
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

# Poisson Surface Reconstruction 메쉬 생성
mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

area = mesh1.get_surface_area()
print("Surface area of mesh1:", area)


