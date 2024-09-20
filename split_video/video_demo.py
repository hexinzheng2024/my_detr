# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import os

# Add the directory containing module_A to sys.path
sys.path.append('../')

from mmdet.apis import inference_detector, init_detector
from projects import *
from mmdet.core.evaluation.class_names import DatasetEnum


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device,dataset=DatasetEnum.LVIS)

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    frame_count = 0
    output_image_dir = './output/images'
    output_label_dir = './output/labels'
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    for frame in mmcv.track_iter_progress(video_reader):
        frame_count += 1
        if frame_count % 10 != 0:
            continue
        else:
            result = inference_detector(model, frame)  
            frame_output = model.show_result(frame, result, score_thr=args.score_thr)    
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            output_image_path = os.path.join(output_image_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(output_image_path, frame)        



            label_content = []
            for i in range(bboxes.shape[0]):
                if bboxes[i, 4] > args.score_thr:
                    label_content.append(str(bboxes[i].tolist()))

            output_label_path = os.path.join(output_label_dir, f'frame_{frame_count:06d}.txt')
            with open(output_label_path, "w", encoding="utf-8") as file:
                file.write("\n".join(label_content))
            # frame = model.show_result(frame, result, score_thr=args.score_thr)
            output_image_path = './output_image.jpg'  # 指定保存路径和文件名
            cv2.imwrite(output_image_path, frame_output)
            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', args.wait_time)
            if args.out:
                video_writer.write(frame_output)
            # break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
