import cv2
import yaml
import json
import mmcv
import os,sys
import base64
from termcolor import colored

# Add the directory containing module_A to sys.path
sys.path.append('../')

from mmdet.apis import inference_detector, init_detector
from projects import *
from mmdet.core.evaluation.class_names import DatasetEnum

class Video_Trans:

    def __init__(self):
        self.output_image_dir = ''
        self.output_label_dir = ''
        self.output_video_file = ''
        self.output_debug_image_path = './output_debug_image.jpg'
        self.config = []

    def clear_folder(self, folder_path):
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 删除文件
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # 如果是空的文件夹，删除它
            except Exception as e:
                print(f"删除文件 {file_path} 时发生错误: {e}")

    def init_param(self):
        self.config = self.read_param_from_yaml()

        # create images / labels / output_video
        self.output_image_dir = os.path.join(self.config['output_dir'], self.config['output_images'])
        self.output_label_dir = os.path.join(self.config['output_dir'], self.config['output_labels'])
        self.output_label_removed_dir = os.path.join(self.config['output_dir'], self.config['output_labels_removed'])
        self.output_crop_image_dir = os.path.join(self.config['output_dir'], self.config['output_crop_images'])
        self.output_crop_image_removed_dir = os.path.join(self.config['output_dir'], self.config['output_crop_images_removed'])

        self.output_jsonl_file = os.path.join(self.config['output_dir'], self.config['output_jsonl'])
        self.output_video_file = os.path.join(self.config['output_dir'], self.config['output_video'])

        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)
        os.makedirs(self.output_label_removed_dir, exist_ok=True)
        os.makedirs(self.output_crop_image_dir, exist_ok=True)
        os.makedirs(self.output_crop_image_removed_dir, exist_ok=True)

        self.clear_folder(self.output_crop_image_dir)
        self.clear_folder(self.output_label_dir)
        self.clear_folder(self.output_label_removed_dir)
        self.clear_folder(self.output_crop_image_dir)
        self.clear_folder(self.output_crop_image_removed_dir)
        if (os.path.exists(self.output_jsonl_file)):
            os.remove(self.output_jsonl_file)


        return self.config

    def read_param_from_yaml(self, yaml_file = 'config.yaml'):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def write_param_to_yaml(self,config_data, yaml_file = 'config.yaml'):
        with open(yaml_file, "w") as file:
            yaml.safe_dump(config_data, file, default_flow_style=False, allow_unicode=True)

    def init_detector(self):
        self.model = init_detector(self.config['lvis'], self.config['model'], device=self.config['device'],dataset=DatasetEnum.LVIS)
        return self.model

    def get_video_reader_and_writer(self):
        video_reader = mmcv.VideoReader(self.config['video'])
        video_writer = None
        if self.config['output_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.output_video_file, fourcc, video_reader.fps,
                (video_reader.width, video_reader.height))

        self.video_reader = video_reader
        self.video_writer = video_writer

        return video_reader

    def output_image_path(self, frame_count):
        return os.path.join(self.output_image_dir, f'frame_{frame_count:06d}.jpg')

    def output_label_path(self, frame_count):
        return os.path.join(self.output_label_dir, f'frame_{frame_count:06d}.txt')

    def output_label_removed_path(self, frame_count):
        return os.path.join(self.output_label_removed_dir, f'frame_r_{frame_count:06d}.txt')

    def _write_labels(self, frame_count, bboxes):
        label_content = []

        rows_num = bboxes.shape[0]
        for i in range(rows_num):
            if bboxes[i, 4] > self.config['score_thr']:
                label_content.append(" ".join(map(str, bboxes[i])))

        output_label_path = self.output_label_path(frame_count)

        # print(f'\n\nlabel_content = {label_content}')
        with open(output_label_path, "w", encoding="utf-8") as file:
            file.write("\n".join(label_content))

    def _write_images(self, frame_count, frame, frame_output):
        cv2.imwrite(self.output_image_path(frame_count), frame)
        output_image_path = self.output_debug_image_path
        cv2.imwrite(output_image_path, frame_output)

    def _write_video(self, frame_output):
        if self.config['output_video']:
            self.video_writer.write(frame_output)


    def output_percent(self, frame_count, total_frames):
        percentage = (frame_count / total_frames) * 100
        sys.stdout.write(f"\rProcessing video: {percentage:.2f}% completed")
        sys.stdout.flush()

    def get_bboxes(self, model, frame):
        # Inference image(s) with the detector
        result = inference_detector(model, frame)
        frame_output = model.show_result(frame, result, score_thr=self.config['score_thr'])

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        return bboxes, frame_output

    def write_info(self, bboxes,frame_count, frame, frame_output):
        self._write_labels(frame_count, bboxes)
        self._write_images(frame_count, frame, frame_output)
        self._write_video(frame_output)

    def release_video_handler(self):
        if self.video_writer:
            self.video_writer.release()

    # overlap的条件太强了，有很多正确的分类被移除
    def rule_overlap(self,box, i, j, to_remove):
        # box = [x1, y1, x2, y2]
        x1, y1, x2, y2 = box[i]
        x3, y3, x4, y4 = box[j]
        
        # Calculate the (x, y) coordinates of the intersection rectangle
        x13 = max(x1, x3)
        y13 = max(y1, y3)
        x24 = min(x2, x4)
        y24 = min(y2, y4)
        # Calculate the area of intersection rectangle
        inter_area = max(0, x24 - x13) * max(0, y24 - y13)
        
        # Calculate the area of both the bounding boxes
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        # Calculate the overlap ratio for each box
        ratio1 = inter_area / box1_area
        ratio2 = inter_area / box2_area

        if ratio1 > 0.9 or ratio2 > 0.9:
            if box1_area < box2_area:
                to_remove.add(i)
                print(colored(f'box1_area = {box1_area}\tbox2_area = {box2_area}\tinner_area = {inter_area}','cyan'))
                print(colored(f'ratio1 : {ratio1} \t ratio2 : {ratio2}', 'cyan'))
                print(colored(f'rule_ovelap {i} first remove', 'red'))
                return 'remove first'
            else:
                to_remove.add(j)
                print(colored(f'box1_area = {box1_area}\tbox2_area = {box2_area}\tinner_area = {inter_area}','cyan'))
                print(colored(f'ratio1 : {ratio1} \t ratio2 : {ratio2}', 'cyan'))
                print(colored(f'rule_ovelap {j} second remove', 'red'))
                return 'remove second'
        return ''
    
    def rule_min_square(self, box, i, to_remove):
        # box = [x1, y1, x2, y2]
        # print(colored(f'box : {box}', 'red'))
        x1, y1, x2, y2 = box[i]

        w = round(x2 - x1, 2)
        h = round(y2 - y1, 2)

        square = round(w * h, 2)

        if square < 40 * 40 : 
            print(colored(f'rule_min_square {i} ==> w : {w} \t h : {h}', 'red'))
            to_remove.add(i)
            return True
        else:
            return False

    def rule_max_square(self, box, i, to_remove):
        # box = [x1, y1, x2, y2]
        # print(colored(f'box : {box}', 'red'))
        x1, y1, x2, y2 = box[i]

        w = round(x2 - x1, 2)
        h = round(y2 - y1, 2)

        square = round(w * h, 2)

        if square > 400*400 : 
            print(colored(f'rule_max_square {i} ==> w : {w} \t h : {h}', 'red'))
            to_remove.add(i)
            return True
        else:
            return False

    def process_overlaped_labels_file(self, frame_count):

        label_file_path = self.output_label_path(frame_count)
        label_removed_file_path = self.output_label_removed_path(frame_count)

        with open(label_file_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            elements = line.strip().split(' ')
            box = list(map(float, elements[:4]))
            boxes.append(box)
        
        to_remove = set()

        for i in range(len(boxes)):
            if self.rule_max_square(boxes, i, to_remove) :
                to_remove.add(i)

        for i in range(len(boxes)):
            if (i in to_remove):
                continue
            if self.rule_min_square(boxes, i, to_remove) :
                continue
            # for j in range(i + 1, len(boxes)):
            #     if self.rule_overlap(boxes, i, j, to_remove) == 'remove first' :
            #         print(colored('skip remain loop', 'green'))
            #         break

        print(colored(f'removes = {len(to_remove)}', 'blue'))
        print(colored(f'accepts = {len(boxes) - len(to_remove)}', 'blue'))
        
        # Write the filtered boxes back to the file
        with open(label_file_path, 'w') as f:
            for idx, line in enumerate(lines):
                if idx not in to_remove:
                    f.write(line)

        with open(label_removed_file_path, 'w') as f:
            for idx, line in enumerate(lines):
                if idx in to_remove:
                    f.write(line)
        
        return to_remove
    
    def extract_clip_from_labels(self, label_file):
        """
        Extract xywh and line numbers from a YOLO bounding box txt file.

        Parameters:
        - txt_path: str, path to the txt file

        Returns:
        - xy_list: list of tuples, each tuple contains (line_number, (x, y, w, h))
        """

        xy_list = []

        # Open and read the txt file
        with open(label_file, 'r') as file:
            lines = file.readlines()

            # Process each line
            for line_number, line in enumerate(lines, start=1):
                # if line.startswith('0') :
                parts = line.strip().split(' ')
                if len(parts) == 5:  # YOLO format: class_id x_center y_center width height
                    x1, y1, x2, y2 = map(float, parts[:4])  # Convert the coordinates to float
                    xy_list.append([line_number, x1, y1, x2, y2])

        return xy_list

    def convert_image_to_base64(self, image):
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

    def save_cropped_image(self, image_path, cropped_image, line_num, output_cropped_image_dir):
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
        base_image_name = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(base_image_name)
        new_image_name = f"{image_name}_line_{line_num}{image_ext}"
        new_image_path = os.path.join(output_cropped_image_dir, new_image_name)

        # Save the cropped image
        cv2.imwrite(new_image_path, cropped_image)
        return new_image_name

    def crop_image_by_bbox(self, label_file, src_image_file, cropped_image_dir):
        """
        Crop an image based on a bounding box.

        Parameters:
        - image_path: str, path to the image file
        - bbox: tuple, bounding box in the format (line_num, x, y, width, height)

        Returns:
        - cropped_image: numpy array, the cropped image
        - line_num: int, the line number from the bounding box
        """
        xy_list = self.extract_clip_from_labels(label_file)

        # Read the image using OpenCV
        cropped_images = []
        for bboxes in xy_list:
            # Read the image using OpenCV
            image = cv2.imread(src_image_file)
            if image is None:
                raise ValueError(f"Failed to load image {src_image_file}")

            line_num, x1, y1, x2, y2 = bboxes
            # print(colored(f'bbox = {bboxes}', 'red'))
            cropped_image = image[ int(y1) : int(y2) ,int(x1) : int(x2)]
            encoded_string = self.convert_image_to_base64(cropped_image)
            new_image_name = self.save_cropped_image(src_image_file, cropped_image, line_num, cropped_image_dir)
            cropped_images.append((new_image_name, encoded_string))

        return cropped_images

    def write_jsonl_file(self, data_array):
        json_file = self.output_jsonl_file
        with open(json_file, 'w') as file:
            for item in data_array:
                custom_id, content = item
                data = {
                    "custom_id":custom_id,
                    "method":"POST",
                    "url":"/v1/chat/completions",
                    "body":{
                        "model":"gpt-4o",
                        "messages":[
                            {
                                "role":"system",
                                "content":"You are a bottle recognition assistant."
                            },
                            {
                                "role":"user",
                                "content":[
                                {
                                    "type":"text",
                                    "text":"The image shows a recycled bottle. If it is a beverage bottle, please answer whether the bottle has a label with 'yes' or 'no'. If it is a glass bottle, ignore the label and the liquid inside, and answer the color of the glass bottle with 'brown', 'green', or 'clear'. If none of these conditions are met, answer 'others'."
                                },
                                {
                                    "type":"image_url",
                                    "image_url":{
                                        "url":f"data:image/jpeg;base64,{content}"
                                    }
                                }
                                ]
                            }
                        ],
                        "max_tokens":1000
                    }
                }
                file.write(json.dumps(data) + '\n')


def main():

    obj = Video_Trans()

    # 读取配置文件获得初始化参数
    obj.init_param()

    # 获取模型
    model = obj.init_detector()

    # 获取 video_reader 和 video_writer
    video_reader = obj.get_video_reader_and_writer()

    FRAME_SKIP = 10

    try:
        frame_count = 0
        for frame in mmcv.track_iter_progress(video_reader):
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            else:
                # 获得 bboxes 和 frame_output
                bboxes, frame_output = obj.get_bboxes(model, frame)

                # 将信息写入到输出labels等文件中
                obj.write_info(bboxes, frame_count, frame, frame_output)

                # 处理labels文件
                obj.process_overlaped_labels_file(frame_count)

                # crop normal images
                base64image = obj.crop_image_by_bbox(
                    obj.output_label_path(frame_count), 
                    obj.output_image_path(frame_count),
                    obj.output_crop_image_dir)

                # write OpenAI api
                obj.write_jsonl_file(base64image)

                # crop removed images
                obj.crop_image_by_bbox(
                    obj.output_label_removed_path(frame_count), 
                    obj.output_image_path(frame_count),
                    obj.output_crop_image_removed_dir
                    )

        obj.release_video_handler()

    except KeyboardInterrupt:
        print('\nprogram end by user.')

if __name__ == '__main__':
    main()
