from depth.depth_area import calculate_surface_area 
from segmentation.inference.utils import segmented_area



if __name__ == "__main__":
    image_path =""
    pointcloud_path = "depth/depth_wrinkle.json"
    depth_coordinates = segmented_area(image_path)
    print("depth_coordinates is : ", len(depth_coordinates))
    area_3d = calculate_surface_area(depth_coordinates, pointcloud_path)
    print(area_3d)