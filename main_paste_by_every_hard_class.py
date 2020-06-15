#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/12
# @Author  : shawn_zhu
# @Site    : 
# @File    : main_paste_by_every_hard_class.py
# @Software: PyCharm

"""
import os
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np

from aug import ensure_dir_exists, img_paths2label_paths, paste_small_objects_to_single_img

# 全数据集
# IMG_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\JPEGImages'
# ANNO_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Annotations'

# 训练集
IMG_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\训练集和验证集\训练集\Images'
ANNO_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\训练集和验证集\训练集\Annotation'

PASTE_IMG_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Augmentation\Images'
PASTE_ANNO_DIR = r'G:\比赛\华为垃圾目标检测分类\数据\trainval\VOC2007\Augmentation\Annotations'

# TODO: cv2不支持中文路径，这里可以check一下
CROPED_IMG_DIR = r'G:\data\trainval\VOC2007\Crop_3_150x150'

MAX_WIDTH = 1000
MAX_HEIGHT = 1000


def search_anno_dir(anno_dir, related_classes):
    """
            将尺寸大于一定程度的图像选择出来
    :param related_classes:
    :param anno_dir:
    :return:    返回item的id
    """
    result = list()

    anno_xml_files = os.listdir(anno_dir)
    anno_xml_file_paths = [os.path.join(anno_dir, x) for x in anno_xml_files]

    for anno_path in tqdm(anno_xml_file_paths, desc='searching for big resolution image'):
        # TODO: 选择特定的对象增强
        size, use = read_anno_for_size_cls(anno_path, related_classes)

        # 不符合类别，就跳过
        if not use:
            continue

        # 判断标注尺寸
        if size[0] >= MAX_HEIGHT and size[1] >= MAX_WIDTH:
            result.append(os.path.basename(anno_path).split('.xml')[0])

    return result


def read_anno_for_size_cls(anno_path, related_classes):
    """
        读取annotation，返回对应的值

    :param related_classes:
    :param anno_path:
    :return:
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # 收集是否包含了某些需要的类别，为特定类别的融合做准备
    use_this_sample = False
    for obj in root.iter('object'):
        # 统计类别数
        name = obj.find('name').text
        if name in related_classes:
            use_this_sample = True
            break

    return (h, w), use_this_sample

def get_all_croped_image_path(_hard_cls, croped_dir=CROPED_IMG_DIR):
    """
        获取crop图像的列表，返回dict
    :param _hard_cls:
    :param hard_cls:
    :param croped_dir:
    :return:
    """
    result_dict = dict()

    # 文件格式: [cls]_[source]_[timestamp].jpg ---> 文件格式: [cls]_[source]_[timestamp]
    img_file_names = [x.split('.jpg')[0] for x in os.listdir(croped_dir)]

    for file_name in img_file_names:
        _cls = file_name.split('_')[0]

        # TODO: 特定类别的crop图像
        if _cls != _hard_cls:
            continue

        result_dict[file_name] = _cls

    return result_dict


if __name__ == "__main__":
    """
    书籍纸张: ['充电宝', '塑料器皿', '塑料玩具', '快递纸袋', '插头电线', '污损塑料', '洗护用品', '玻璃器皿', '纸盒纸箱', '软膏', '过期药物', '锅', '陶瓷器皿', '食用油桶']
    污损用纸: ['一次性快餐盒', '剩饭剩菜', '垃圾桶', '塑料器皿', '塑料玩具', '大骨头', '易拉罐', '果皮果肉', '污损塑料', '洗护用品', '玻璃器皿', '筷子', '纸盒纸箱', '菜帮菜叶', '蛋壳', '金属厨具', '金属器皿', '陶瓷器皿', '饮料瓶', '鱼骨']
    金属器皿: ['一次性快餐盒', '剩饭剩菜', '塑料器皿', '大骨头', '插头电线', '易拉罐', '果皮果肉', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '玻璃器皿', '砧板', '筷子', '纸盒纸箱', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '酒瓶', '金属厨具', '金属食品罐', '锅', '陶瓷器皿', '饮料瓶', '鱼骨']
    """
    hard_cls = ['书籍纸张', '污损用纸', '金属器皿']
    related_cls = [
        ['充电宝', '塑料器皿', '塑料玩具', '快递纸袋', '插头电线', '污损塑料', '洗护用品', '玻璃器皿', '纸盒纸箱', '软膏', '过期药物', '锅', '陶瓷器皿', '食用油桶'],
        ['一次性快餐盒', '剩饭剩菜', '垃圾桶', '塑料器皿', '塑料玩具', '大骨头', '易拉罐', '果皮果肉', '污损塑料', '洗护用品', '玻璃器皿', '筷子', '纸盒纸箱', '菜帮菜叶',
         '蛋壳', '金属厨具', '金属器皿', '陶瓷器皿', '饮料瓶', '鱼骨'],
        ['一次性快餐盒', '剩饭剩菜', '塑料器皿', '大骨头', '插头电线', '易拉罐', '果皮果肉', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '玻璃器皿', '砧板', '筷子',
         '纸盒纸箱', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '酒瓶', '金属厨具', '金属食品罐', '锅', '陶瓷器皿', '饮料瓶', '鱼骨']
    ]
    for i in range(len(hard_cls)):
        # 待贴图的样本类别
        cls = hard_cls[i]
        # 与其相关的样本类别
        rlt_cls = related_cls[i]

        # 获取可以贴图的样本
        source_ids = search_anno_dir(ANNO_DIR, related_classes=rlt_cls)

        # step 1 获取所有的crop下来的图像
        croped_image_dict = get_all_croped_image_path(cls, CROPED_IMG_DIR)

        # 遍历每一个图像，并且在里面随机进行paste
        for source_id in tqdm(source_ids, desc=f'hard class:{cls}, pasting croped images.'):
            source_img_path = os.path.join(IMG_DIR, source_id+'.jpg')
            source_anno_path = os.path.join(ANNO_DIR, source_id+'.xml')

            # step 2 随机从crop的图像中选取n个出来paste，个数取决于这个图像的分辨率, 将得到的图像保存的熬指定的位置
            paste_small_objects_to_single_img(source_img_path, source_anno_path, croped_image_dict,
                                                      croped_dir=CROPED_IMG_DIR,
                                                      save_img_dir=PASTE_IMG_DIR,
                                                      save_anno_dir=PASTE_ANNO_DIR,
                                                      prob=0.3,
                                                      origin_rescale=True,  # 对于原始图像resize
                                                      croped_rescale=True)  # 对于crop的图像resize