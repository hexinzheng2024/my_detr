from PIL import Image, ImageDraw, ImageFont
import os
def draw_bounding_box(image, boxes):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 使用自定义字体，设置字体大小
    font = ImageFont.truetype("../fonts/arial.ttf", 36)  # 确保系统有该字体（可以更改字体路径）

    for box in boxes:
        line_number, label, x_center, y_center, w, h, scores = box
        # Convert from normalized coordinates to absolute pixel values
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        # Calculate the bounding box corners
        left = x_center - w / 2
        top = y_center - h / 2
        right = x_center + w / 2
        bottom = y_center + h / 2

        # Draw the bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the label (class name and line number) above the bounding box
        text_position = (left, top - 30)  # 标签位置
        text = f"{line_number}: {label}"  # 显示行号和类别标签
        draw.text(text_position, text, fill="red", font=font)  # 使用大号字体绘制标签

    return image

def process_txt_file(txt_file_path, img_file_path, output_file_path):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for idx, line in enumerate(lines):
        elements = line.strip().split()
        label = 'none_name'  # 读取类别标签
        box = list(map(float, elements[:]))  # 跳过标签并处理box
        boxes.append((idx + 1, label, *box))  # 保存行号和类别标签

    image = Image.open(img_file_path)
    image_with_boxes = draw_bounding_box(image, boxes)
    image_with_boxes.save(output_file_path)

def process_all_files(labels_folder, images_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for txt_filename in os.listdir(labels_folder):
        if txt_filename.endswith('.txt'):
            txt_file_path = os.path.join(labels_folder, txt_filename)
            img_filename = txt_filename.replace('.txt', '.jpg')
            img_file_path = os.path.join(images_folder, img_filename)
            output_file_path = os.path.join(output_folder, img_filename)
            
            if os.path.exists(img_file_path):
                process_txt_file(txt_file_path, img_file_path, output_file_path)

# 调用函数处理所有文件
labels_folder = '/home/eii/Downloads/test/labels'  # 替换为您的labels文件夹路径
images_folder = '/home/eii/Downloads/test/images'  # 替换为您的images文件夹路径
output_folder = '/home/eii/Downloads/test/output_2'  # 替换为您想要保存输出图片的文件夹路径
process_all_files(labels_folder, images_folder, output_folder)
