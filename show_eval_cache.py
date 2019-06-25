# coding=utf-8
import numpy as np
import torch
import cv2
import os
import argparse
from openpifpaf.network import nets
from openpifpaf import decoder, show, transforms
import ujson

COCO_PERSON_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]


def plot_points(img, labels, skeleton):

    point_color = (255, 255, 255)  # 白点
    line_color = (0, 0, 255)       # 红线
    point_size = 1
    thickness = 1

    if labels.shape[0] == 0:
        # pred 没检测出来, 那就return吧
        return
    nums, kth = labels.shape[:2]
    for i in range(nums):
        # r = np.random.randint(0, 255)  # 可以考虑自己随机颜色
        # g = np.random.randint(0, 255)
        # b = np.random.randint(0, 255)
        # color = (r, g, b)
        label = labels[i, :, :]
        x = []
        for j in range(kth):
            if label[j][-1] >= 0.7:
                x.append((label[j][0], label[j][1], j+1))   # j+1代表这个点的index, 后续点连线用到

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
    parser.add_argument('--disable_cuda', action='store_true', default=None,
                        help='disable CUDA')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        print('************************ using gpu *****************************')
        args.device = torch.device('cuda')

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)

    # own coco val json
    f = open('/home/chenjia/pedestrian_det/chenjiaPifPaf/splitjson/test_5000.json')
    js = ujson.load(f)
    img_paths = js['images']   # len==5000, img的相对路径
    img_path_root = '/data/nfs_share/public/zxyang/human_data/detection_data_cpu'
    out_root = '/home/chenjia/pedestrian_det/openpifpaf/show_eval'

    # random check pred result
    for i in range(50):
        print('*************** run the ', i + 1, 'image ******************')
        ind = np.random.randint(0, 4999)
        img_path = os.path.join(img_path_root, img_paths[ind]['file_name'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (683, 384))

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_image_cpu = transforms.image_transform(image.copy())   # normalize

        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)  # transpose 2,0,1
        fields = processor.fields(torch.unsqueeze(processed_image, 0))[0]
        keypoint_sets, _ = processor.keypoint_sets(fields)

        # plot pred result
        plot_points(image, keypoint_sets, COCO_PERSON_SKELETON)
        cv2.imwrite(os.path.join(out_root, str(ind) + '.jpg'), image)




if __name__ == '__main__':
    main()