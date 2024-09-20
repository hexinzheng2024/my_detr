import os

def overlap_ratio(box1, box2):
    # box = [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y) coordinates of the intersection rectangle
    left1, top1 = x1 - w1 / 2, y1 - h1 / 2
    right1, bottom1 = x1 + w1 / 2, y1 + h1 / 2
    left2, top2 = x2 - w2 / 2, y2 - h2 / 2
    right2, bottom2 = x2 + w2 / 2, y2 + h2 / 2
    
    # Calculate the (x, y) coordinates of the intersection rectangle
    xi1 = max(left1, left2)
    yi1 = max(top1, top2)
    xi2 = min(right1, right2)
    yi2 = min(bottom1, bottom2)
    # Calculate the area of intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate the area of both the bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate the overlap ratio for each box
    ratio1 = inter_area / box1_area
    ratio2 = inter_area / box2_area
    
    return ratio1, ratio2

def process_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        # elements = line.strip().split()
        elements = line.strip()[1:-1].split(',')
        box = list(map(float, elements[1:]))
        boxes.append(box)
    
    to_remove = set()
    
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ratio1, ratio2 = overlap_ratio(boxes[i], boxes[j])
            if ratio1 > 0.9 or ratio2 > 0.9:
                area_i = boxes[i][2] * boxes[i][3]
                area_j = boxes[j][2] * boxes[j][3]
                if area_i > area_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    # print(to_remove)
    
    # Write the filtered boxes back to the file
    with open(file_path, 'w') as f:
        for idx, line in enumerate(lines):
            if idx not in to_remove:
                f.write(line)

def process_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
        # if filename == 'conveyor-pet1-frame_002360.txt':
            process_txt_file(os.path.join(folder_path, filename))

# 调用函数处理所有文件
# folder_path = 'labels'  # 替换为您的labels文件夹路径
folder_path = 'output/labels'
process_all_files_in_folder(folder_path)
