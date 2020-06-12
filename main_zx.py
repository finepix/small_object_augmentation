#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/11
# @Author  : shawn_zhu
# @Site    : 
# @File    : main_zx.py
# @Software: PyCharm

"""
import os
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np

from aug import ensure_dir_exists, img_paths2label_paths, paste_small_objects_to_single_img


IMG_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\JPEGImages'
ANNO_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Annotations'
PASTE_IMG_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Augmentation\Images'
PASTE_ANNO_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Augmentation\Annotations'

# TODO: cv2不支持中文路径，这里可以check一下
CROPED_IMG_DIR = r'G:\data\trainval\VOC2007\Crop_3_150x150'

MAX_WIDTH = 800
MAX_HEIGHT = 800


def search_anno_dir(anno_dir):
    """
            将尺寸大于一定程度的图像选择出来
    :param anno_dir:
    :return:    返回item的id
    """
    result = list()

    anno_xml_files = os.listdir(anno_dir)
    anno_xml_file_paths = [os.path.join(anno_dir, x) for x in anno_xml_files]

    for anno_path in tqdm(anno_xml_file_paths, desc='searching for big resolution image'):
        size = read_anno_for_size(anno_path)

        # 判断标注尺寸
        if size[0] >= MAX_HEIGHT and size[1] >= MAX_WIDTH:
            result.append(os.path.basename(anno_path).split('.xml')[0])

    return result


def read_anno_for_size(anno_path):
    """
        读取annotation，返回对应的值

    :param anno_path:
    :return:
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # TODO: 收集是否包含了某些需要的类别，为特定类别的融合做准备

    return h, w

def get_all_croped_image_path(croped_dir=CROPED_IMG_DIR):
    """
        获取crop图像的列表，返回dict
    :param croped_dir:
    :return:
    """
    result_dict = dict()

    # 文件格式: [cls]_[source]_[timestamp].jpg ---> 文件格式: [cls]_[source]_[timestamp]
    img_file_names = [x.split('.jpg')[0] for x in os.listdir(croped_dir)]

    for file_name in img_file_names:
        cls = file_name.split('_')[0]
        result_dict[file_name] = cls

    return result_dict


if __name__ == "__main__":
    source_ids = search_anno_dir(ANNO_DIR)
    print(source_ids)

    # TODO: step 1 获取所有的crop下来的图像
    croped_image_dict = get_all_croped_image_path(CROPED_IMG_DIR)

    # 遍历每一个图像，并且在里面随机进行paste
    for source_id in tqdm(source_ids, desc='pasting croped images.'):
        source_img_path = os.path.join(IMG_DIR, source_id+'.jpg')
        source_anno_path = os.path.join(ANNO_DIR, source_id+'.xml')

        # TODO: step 2 随机从crop的图像中选取n个出来paste，个数取决于这个图像的分辨率
        paste_small_objects_to_single_img(source_img_path, source_anno_path, croped_image_dict,
                                                  croped_dir=CROPED_IMG_DIR,
                                                  save_img_dir=PASTE_IMG_DIR,
                                                  save_anno_dir=PASTE_ANNO_DIR)

        # TODO: step 3 将得到的图像保存的熬指定的位置






































