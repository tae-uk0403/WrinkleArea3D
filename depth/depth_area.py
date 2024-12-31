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
        for i in range(len(depth_json["Depth"])):
            x = i % depth_w
            y = i // depth_w
            camera_depth_array[x][y] = depth_json["Depth"][i]

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
    o3d.io.write_point_cloud(
        output_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_true))
    )

    return pcd_true


def create_seg_pcd(depth_coordinates, pcd_true):
    # depth_coordinates를 create_segmentation_array와 유사한 형태로 변환
    seg_array = [
        [coord[0], coord[1]] for coord in depth_coordinates
    ]  # 각 좌표를 직접 추가

    pcd_segmented = []
    for i in range(len(seg_array)):
        [seg_x, seg_y] = seg_array[i]
        pcd_segmented.append(pcd_true[seg_y * 192 + 192 - seg_x])

    pcd_segmented = np.array(pcd_segmented, dtype=np.float32)
    return pcd_segmented  # 변환된 segmentation_array 반환


def calculate_surface_area(depth_coordinates, pointcloud_path):

    full_pcd = depth_to_ply(pointcloud_path)
    segmented_pcd = create_seg_pcd(depth_coordinates, full_pcd)
    area = area_from_mesh(full_pcd, segmented_pcd, threshold=0.01)

    # segmented_output_path = "segmented_output.ply"  # 저장할 파일 경로
    # o3d.io.write_point_cloud(
    #     segmented_output_path,
    #     o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_segmented)),
    # )
    # print("pcd_segmented is ", pcd_segmented)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcd_segmented)
    # # 노멀(법선 벡터) 계산
    # pcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20)
    # )

    # # Poisson Surface Reconstruction 메쉬 생성
    # mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     pcd, depth=9
    # )

    # area = mesh1.get_surface_area()
    # print("area is ", area)
    return area


def area_from_mesh(full_pcd, segmented_pcd, threshold=0.01):
    """
    전체 포인트 클라우드에서 메쉬를 생성하고, Segmentation 영역을 기반으로 서브 메쉬를 추출합니다.

    Parameters:
        full_pcd (o3d.geometry.PointCloud): 전체 포인트 클라우드
        segmented_pcd (o3d.geometry.PointCloud): Segmentation 포인트 클라우드
        threshold (float): 근접 거리 임계값 (디폴트: 0.01)

    Returns:
        mesh (o3d.geometry.TriangleMesh): 전체 메쉬
        seg_mesh (o3d.geometry.TriangleMesh): Segmentation 서브 메쉬
        seg_area (float): Segmentation 서브 메쉬의 표면적
    """

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(full_pcd.astype(np.float64))

    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(segmented_pcd.astype(np.float64))

    # 1. 전체 포인트 클라우드의 노멀 추정
    pcd_full.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # 2. 전체 메쉬 생성 (Poisson)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_full, depth=9
    )

    # 3. Segmentation 영역 포인트 KD-Tree 생성
    segmented_tree = o3d.geometry.KDTreeFlann(pcd_seg)

    # 4. Segmentation에 포함된 정점만 추출
    vertices = np.asarray(mesh.vertices)
    num_vertices = vertices.shape[0]

    mask = np.zeros(num_vertices, dtype=bool)
    for i, vertex in enumerate(vertices):
        [k, idx, dist] = segmented_tree.search_knn_vector_3d(vertex, 1)
        if k > 0 and dist[0] < (threshold * threshold):
            mask[i] = True

    # 5. 삼각형 필터링
    triangles = np.asarray(mesh.triangles)
    triangle_mask = mask[triangles].all(axis=1)
    seg_triangles = triangles[triangle_mask]

    # 6. 정점 재매핑
    seg_indices = np.unique(seg_triangles)
    new_index = np.full(num_vertices, fill_value=-1, dtype=int)
    new_index[seg_indices] = np.arange(len(seg_indices))

    new_vertices = vertices[seg_indices]
    new_triangles = np.array([[new_index[idx] for idx in tri] for tri in seg_triangles])

    # 7. 서브 메쉬 생성 및 면적 계산
    seg_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(new_vertices),
        o3d.utility.Vector3iVector(new_triangles),
    )
    seg_mesh.compute_vertex_normals()
    seg_area = seg_mesh.get_surface_area()

    return seg_area
