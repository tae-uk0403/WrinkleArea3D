from utils import preprocess_image_for_test, load_model, segmented_area, convert_to_depth_coordinates

image_tensor = preprocess_image_for_test(image_path, target_size=(512, 512))