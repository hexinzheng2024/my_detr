import base64
import json
import os
import cv2

class utils:

    def process_images_and_labels(self, image_filename, labels_filename):
        results = {}

        if os.path.exists(labels_filename) & os.path.exists(image_filename):
            result = self.extract_xywh_from_yolo_txt(labels_filename)
            results[image_filename] = result

        return results

    def _extract_xywh_from_yolo_txt(self, label_filename):
        """
        Extract xywh and line numbers from a YOLO bounding box txt file.

        Parameters:
        - txt_path: str, path to the txt file

        Returns:
        - xywh_list: list of tuples, each tuple contains (line_number, (x, y, w, h))
        """
        xywh_list = []

        # Open and read the txt file
        with open(label_filename, 'r') as file:
            lines = file.readlines()

            # Process each line
            for line_number, line in enumerate(lines, start=1):
                # if line.startswith('0') :
                parts = line.strip()[1:-1].split(' ')
                if len(parts) == 5:  # YOLO format: class_id x_center y_center width height
                    x, y, w, h = map(float, parts[:4])  # Convert the coordinates to float
                    xywh_list.append([line_number, x, y, w, h])

        return xywh_list