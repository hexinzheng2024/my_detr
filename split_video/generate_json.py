import base64
import json
import os
import cv2

def convert_image_to_base64(image):
    """
    Convert a cropped image to a base64 encoded string.
    
    Parameters:
    - image: numpy array, the cropped image

    Returns:
    - encoded_string: str, the base64 encoded string of the image
    """
    if image is None:
        raise ValueError("Failed to load image")

    # Encode the image as a JPEG in memory
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string
    
def crop_image_by_bbox(results, save_dir):
    """
    Crop an image based on a bounding box.

    Parameters:
    - image_path: str, path to the image file
    - bbox: tuple, bounding box in the format (line_num, x, y, width, height)

    Returns:
    - cropped_image: numpy array, the cropped image
    - line_num: int, the line number from the bounding box
    """
    # Read the image using OpenCV
    cropped_images = []
    for image_path, bboxes in results.items():
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image {image_path}")
        # image_height, image_width = 1 , 1

        for bbox in bboxes:
            print(bbox)
            line_num, x1, y1, x2, y2 = bbox
            # print(line_num)
            # Denormalize the bounding box coordinates
            # x = int(x * image_width)
            # y = int(y * image_height)
            # width = int(width * image_width)
            # height = int(height * image_height)

            # # Calculate the top-left corner of the bounding box
            # top_left_x = x - width // 2
            # top_left_y = y - height // 2

            # Crop the image using the bounding box
            # if label == 0 or label == 7:
                
            # cropped_image = image[top_left_y:top_left_y+height, top_left_x:top_left_x+width]
            cropped_image = image[ int(y1) : int(y2) ,int(x1) : int(x2)]
            # print(cropped_image.shape, image_path)
            encoded_string = convert_image_to_base64(cropped_image)
            new_file_name = save_cropped_image(image_path, cropped_image, line_num, save_dir)
            cropped_images.append((new_file_name, encoded_string))

    return cropped_images, line_num

def save_cropped_image(image_path, cropped_image, line_num, save_dir):
    """
    Save the cropped image with a modified filename to a specified directory.

    Parameters:
    - image_path: str, path to the original image file
    - cropped_image: numpy array, the cropped image
    - line_num: int, line number to include in the new file name
    - save_dir: str, directory where the cropped image will be saved

    Returns:
    - new_file_path: str, the path to the saved cropped image
    """
    # Construct the new file name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(base_name)
    new_file_name = f"{file_name}_line_{line_num}{file_ext}"
    new_file_path = os.path.join(save_dir, new_file_name)

    # Save the cropped image
    cv2.imwrite(new_file_path, cropped_image)
    return new_file_name
    
def process_images_and_labels(images_folder, labels_folder):
    results = {}

    # 获取images文件夹中的所有图片文件
    for image_filename in os.listdir(images_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建图片文件的完整路径
            image_path = os.path.join(images_folder, image_filename)

            # 构建对应的txt文件名和路径
            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            txt_path = os.path.join(labels_folder, txt_filename)

            # 如果对应的txt文件存在，处理它
            if os.path.exists(txt_path):
                result = extract_xywh_from_yolo_txt(txt_path)
                results[image_path] = result
                # results.append((image_path, result))

    print(results)
    return results

def extract_xywh_from_yolo_txt(txt_path):
    """
    Extract xywh and line numbers from a YOLO bounding box txt file.

    Parameters:
    - txt_path: str, path to the txt file

    Returns:
    - xywh_list: list of tuples, each tuple contains (line_number, (x, y, w, h))
    """
    xywh_list = []

    # Open and read the txt file
    with open(txt_path, 'r') as file:
        lines = file.readlines()

        # Process each line
        for line_number, line in enumerate(lines, start=1):
            # if line.startswith('0') :
            parts = line.strip().split(' ')
            if len(parts) == 5:  # YOLO format: class_id x_center y_center width height
                x, y, w, h = map(float, parts[:4])  # Convert the coordinates to float
                xywh_list.append([line_number, x, y, w, h])

    return xywh_list

def generate_jsonl_file(data_array, filename="output.jsonl"):
    with open(filename, 'w') as file:
        for item in data_array:
            custom_id, content = item
            data = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a bottle recognition assistant."},
                        {"role": "user", "content": [
                                                    {
                                                    "type": "text",
                                                    "text": "The image shows a recycled bottle. If it is a beverage bottle, please answer whether the bottle has a label with 'yes' or 'no'. If it is a glass bottle, ignore the label and the liquid inside, and answer the color of the glass bottle with 'brown', 'green', or 'clear'. If none of these conditions are met, answer 'others'."
                                                    },
                                                    {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{content}"
                                                    }
                                                    }
                                                ]
    }
                    ],
                    "max_tokens": 1000
                }
            }
            file.write(json.dumps(data) + '\n')

# images_folder = 'images'
# labels_folder = 'labels'
# save_dir = '/media/eii-jh/D/bottle_dataset/done/45/output/testimg_crop'

images_folder = '/home/eii/Downloads/test/images'
labels_folder = '/home/eii/Downloads/test/labels'
save_dir = '/home/eii/Downloads/test/images_crop_1'

results = process_images_and_labels(images_folder, labels_folder)
image,line_num = crop_image_by_bbox(results, save_dir)
generate_jsonl_file(image)