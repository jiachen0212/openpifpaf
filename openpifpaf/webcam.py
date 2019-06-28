"""Webcam demo application.

Example command:
    python -m pifpaf.webcam \
        --checkpoint outputs/resnet101-pif-paf-rsmooth0.5-181121-170119.pkl \
        --src http://128.179.139.21:8080/video \
        --seed-threshold=0.3 \
        --scale 0.2 \
        --connection-method=max
"""

import numpy as np
import argparse
import time
import matplotlib
from collections import Counter  

import matplotlib.pyplot as plt
import torch

import cv2
from .network import nets
from . import decoder, show, transforms

COCO_PERSON_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]


def plot_points(img, labels, skeleton):
    nums, kth = labels.shape[:2]
    for i in range(nums):
        # r = np.random.randint(0, 255)  # 可以考虑自己随机颜色
        # g = np.random.randint(0, 255)
        # b = np.random.randint(0, 255)
        # color = (r, g, b)
        point_color = (255, 255, 255)   # 白点
        line_color = (0, 0, 255)  # 红线
        label = labels[i, :, :]
        x = []
        for j in range(kth):
            if label[j][-1] >= 0.7:
                x.append((label[j][0], label[j][1], j+1))   # j+1代表这个点的index, 后续点连线用到
        point_size =1
        thickness = 1
        for point in x:
            cv2.circle(img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
        for connecte in skeleton:
            for p1 in x:
                for p2 in x:
                    if p1[-1] == connecte[0] and p2[-1] == connecte[1]:
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), line_color, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    parser.add_argument('--no-colored-connections',
                        dest='colored_connections', default=True, action='store_false',
                        help='do not use colored connections to draw poses')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--source', default=0,
                        help='OpenCV source url. Integer for webcams. Or ipwebcam streams.')
    parser.add_argument('--scale', default=0.1, type=float,
                        help='input image scale factor')
    args = parser.parse_args()


    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        print('using gpu ...')
        args.device = torch.device('cuda')

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)
    
    # args.source: 'name1.avi, name2.avi,...'
    avis = []
    counts = Counter(args.source)
    vid_nums = counts[',']  # douhao numbers
    if vid_nums == 0:
        avis = [args.source]
    else:
        for i in range(vid_nums):
            avis.append(args.source.split(',')[i])
        avis.append(args.source.split(',')[-1])
    for vid_id, avi in enumerate(avis):
        print('************* runing the ', avi, 'video *************')
        # name = np.random.randint(1,100)
        videoCapture = cv2.VideoCapture(avi)   # 待检测的video
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        scale = True  # 是否scale视频帧的尺寸
        if scale:
            size = (683, 384)

        ret, frame = videoCapture.read()
        print('whether scale?', scale)
        videoWriter = cv2.VideoWriter('show' + str(vid_id) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        frame_cnt = 0
        while ret:
            frame1 = cv2.resize(frame, size)  # (384, 683, 3)
            image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            processed_image_cpu = transforms.image_transform(image.copy())   # normalize
            processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)  # transpose 2,0,1
            fields = processor.fields(torch.unsqueeze(processed_image, 0))[0]
            keypoint_sets, _ = processor.keypoint_sets(fields)

            if keypoint_sets.shape[0] > 0:
                plot_points(image, keypoint_sets, COCO_PERSON_SKELETON)

            videoWriter.write(image)
            frame_cnt += 1
            print('帧数: ', frame_cnt)
            ret, frame = videoCapture.read()


if __name__ == '__main__':
    main()
