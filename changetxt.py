import os
from PIL import Image

def convert_xyxy_to_xywh(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def process_labels_and_images(labels_dir, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))
            
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                with open(label_path, 'r') as file:
                    lines = file.readlines()

                output_lines = []
                for line in lines:
                    coords = line.strip()[1:-1].split(',')
                    xmin, ymin, xmax, ymax = map(float, coords[:4])
                    x_center, y_center, width, height = convert_xyxy_to_xywh(xmin, ymin, xmax, ymax, img_width, img_height)
                    output_lines.append(f"{x_center} {y_center} {width} {height}\n")

                output_label_path = os.path.join(output_dir, label_file)
                with open(output_label_path, 'w') as output_file:
                    output_file.writelines(output_lines)

# 使用示例
labels_dir = 'output/labels'
images_dir = 'output/images'
output_dir = 'output/labels_out'
process_labels_and_images(labels_dir, images_dir, output_dir)
