#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/10
# @Author  : shawn_zhu
# @Site    : 
# @File    : crop_hard_samples_from_image.py
# @Software: PyCharm

"""


import os
import xml.etree.ElementTree as ET
import cv2
import time
import numpy as np

from tqdm import tqdm


# 针对小目标的增强，参考论文： augmentation for small object detection（https://arxiv.org/pdf/1902.07296.pdf）
# TODO： 先将难分样本的小目标的图像crop出来
# 将尺寸小于200*200的难分样本截取下来

def show_img(_img):
    """
        显示图像
    :param _img:
    :return:
    """
    cv2.imshow('debug', _img)
    cv2.waitKey(0)

def cv_imread(file_path):
    """
        解决cv2读取中文路径的问题

    :param file_path:
    :return:
    """
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), -1)
    return cv_img

def cv_imwrite(filename, src):
    """
        解决cv2不支持中文的问题
    :param filename:
    :param src:
    :return:
    """
    cv2.imencode('.jpg',src)[1].tofile(filename)


MIN_W = 150
MIN_H = 150

ANNO_DIR = r'G:\data\trainval\VOC2007\Annotations'
IMG_DIR = r'G:\data\trainval\VOC2007\JPEGImages'
CROPED_DIR = r'G:\data\trainval\VOC2007\Crop'

xml_files = [os.path.join(ANNO_DIR, x) for x in os.listdir(ANNO_DIR)]

# ap最低的几个类
needed_class = ['书籍纸张', '污损用纸', '金属器皿']

# 1、遍历
needed_xmls = list()
for xml in tqdm(xml_files, desc='searching xml files'):
    tree = ET.parse(xml)
    root = tree.getroot()

    for obj in root.iter('object'):
        # 统计类别数
        name = obj.find('name').text
        if name in needed_class:
            needed_xmls.append(os.path.basename(xml))
            break

# 2、写入文件
with open('txt/hard_samples_3_cat.txt', 'w') as f:
    for item in tqdm(needed_xmls, desc='wirte files'):
        item_id = item.split('.xml')[0]
        f.write(item_id + '\n')

# for xml in needed_xmls:
#     print(xml)

# 3、将图片尺寸小于150*150的截取下来
needed_xml_files = [os.path.join(ANNO_DIR, x) for x in needed_xmls]
for xml in tqdm(needed_xml_files, desc='crop bbox from images'):
    tree = ET.parse(xml)
    root = tree.getroot()

    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in needed_class:
            xml_bbox = obj.find('bndbox')
            box = [int(xml_bbox.find('xmin').text), int(xml_bbox.find('ymin').text),
                   int(xml_bbox.find('xmax').text), int(xml_bbox.find('ymax').text)]
            w = box[2] - box[0]
            h = box[3] - box[1]

            if w <= MIN_W and h <= MIN_H:
                # 图像裁剪
                img_file_name = os.path.basename(xml).replace('.xml', '.jpg')
                img_path = os.path.join(IMG_DIR, img_file_name)

                img = cv2.imread(img_path)
                # debug
                # show_img(img)
                # img2 = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                # show_img(img2)

                # 截取
                img = img[box[1]:box[3], box[0]:box[2]]
                # show_img(img)
                # 文件名
                timestamp = int(time.time() * 1000)
                save_file_name = name + '_' + img_file_name.split('.jpg')[0] + '_' + str(timestamp) + '.jpg'
                save_file_path = os.path.join(CROPED_DIR, save_file_name)

                # cv2.imwrite(save_file_path, img)
                cv_imwrite(save_file_path, img)
