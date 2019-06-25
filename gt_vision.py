# coding=utf-8
import numpy as np
import matplotlib
import os
import sys
import copy
import random
import argparse
import numpy as np
from coco import COCO
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ujson


def plot_points(img, labels):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]
    line_color = (0, 0, 255)    
    for index, label in enumerate(labels):
        x = label['keypoints']
        if len(x) > 0:
            lab = copy.deepcopy(x)
            x = [x[i: i + 2] for i in range(0, len(x), 3)]
            point_size = 1
            point_color = (255, 255, 255) # BGR
            thickness = 4 # 可以为 0 、4、8
            for point in x:
                cv2.circle(img, (point[0], point[1]), point_size, point_color, thickness)
            lab = np.array(lab)
            lab = np.reshape(lab, (17, 3))
            cache = []
            for i, point_ind in enumerate(lab):
                cache.append((point_ind[0], point_ind[1], i + 1))
            for connecte in skeleton:
                for gtp1 in cache:
                    for gtp2 in cache:
                        if gtp1[-1] == connecte[0] and gtp2[-1] == connecte[1]:
                            cv2.line(img, (int(gtp1[0]), int(gtp1[1])), (int(gtp2[0]), int(gtp2[1])), line_color, 1) 


def cv_imread(filepath): #路径中有中文字符的可以使用该方法替代cv2.imread
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img

def draw_box_coco(imagepath, savepath, boxes, img_name, fonts=True):
    #opencv BGR
    vio = (238,130,238) #紫罗兰 (0, 25]
    mag = (255, 0, 255) #洋红色 (25, 75]
    purp = (128,0,128) #紫色 (75, 254)

    cyan = (255, 255, 0) #天蓝色 正常框
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255) #黄色，忽略框
    #font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)

    # im = cv2.imread(imagepath)
    im = cv_imread(imagepath)

    for idx in range(len(boxes)):
        x1 = int(boxes[idx]['bbox'][0])
        y1 = int(boxes[idx]['bbox'][1])
        x2 = int(boxes[idx]['bbox'][0] + boxes[idx]['bbox'][2])
        y2 = int(boxes[idx]['bbox'][1] + boxes[idx]['bbox'][3])
        if boxes[idx]['iscrowd'] == 1:
            _color = yellow
        elif 'hide' in boxes[idx]:
            if boxes[idx]['hide'] > 75:
                _color = purp
            elif boxes[idx]['hide'] > 25:
                _color = mag
            elif boxes[idx]['hide'] > 0:
                _color = vio
            else:
                _color = cyan
        else:
            _color = cyan

        plot_points(im, img_labels)
        # chenjia 0625 not draw bbox
        # cv2.rectangle(img=im, pt1=(x1, y1), pt2=(x2, y2), color=_color, thickness=2)
        
        if fonts:
            info = boxes[idx]['lamp']
            cv2.putText(im, info, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, .6, _color, 1, 2)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    save_f = os.path.join(savepath, img_name + '.jpg')
    cv2.imwrite(save_f, im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='file_path',
                        default='/home/chenjia/pedestrian_det/chenjiaPifPaf/splitjson/test_5000.json')
    parser.add_argument('-r', dest='root_path',default='/data/nfs_share/public/zxyang/human_data/detection_data_cpu')
    parser.add_argument('-n', dest='draw_num', default=20, type=int)  # show 20 imgs
    parser.add_argument('-z', dest='fonts', default=False, type=bool)

    args = parser.parse_args()

    c = COCO(args.file_path)
    print(args.file_path)
    root_path = args.root_path
    save_path = "/home/chenjia/pedestrian_det/openpifpaf/show_eval"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    draw_list = []
    for id, img in c.imgs.items():
        draw_list.append(id)

    if len(draw_list) < args.draw_num:
        draw_num = len(draw_list)
    else:
        draw_num = args.draw_num

    numList = random.sample(draw_list, draw_num)

    for idx, i in enumerate(numList): #random
        img_labels = c.imgToAnns[i]
        img_path = os.path.join(root_path, c.imgs[i]['file_name'])

        draw_box_coco(img_path, save_path, img_labels, str(i), args.fonts)
        
        sys.stdout.write('\r[%d]\tdone in/ %d  query' % (idx, len(numList)))
        sys.stdout.flush()
    print('Done!')