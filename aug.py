import os
import random
import time
from os.path import basename, dirname, join
import cv2
import numpy as np


def random_flip_bbox(roi):
    """
        随机翻转
    :param roi:
    :return:
    """
    # 上下翻转
    if random.randint(0, 1) < 0.5:
        roi = roi[::-1, :, :]
    # 左右翻转
    if random.randint(0, 1) < 0.5:
        roi = roi[:, ::-1, :]

    return roi


def ensure_dir_exists(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        print("Makes new dir:", _dir)


def rescale_labels(labels, img_shape):
    height, width, n_channels = img_shape
    rescaled_boxes = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        x_left = x_c - w * .5
        y_left = y_c - h * .5
        x_right = x_c + w * .5
        y_right = y_c + h * .5
        rescaled_boxes.append([box[0], int(x_left), int(y_left), int(x_right), int(y_right)])
    return rescaled_boxes


def compute_iou(box1, box2):
    cls1, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cls2, b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # Get the coordinates of the intersection rectangle
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    # Compute intersection area
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    # if inter_w <= 0 and inter_h <= 0:  # original: strong condition ?
    if inter_w <= 0 or inter_h <= 0:  # weak condition ?
        return 0
    inter_area = inter_w * inter_h
    # Compute union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def uniform_sample(search_space):
    """Uniformly sample bboxes

    Arguments:
        search_space (4 num) -- range of search

    Returns:
        center of new boxes
    """
    search_x_left, search_y_left, search_x_right, search_y_right = search_space
    new_bbox_x_c = random.randint(search_x_left, search_x_right)
    new_bbox_y_c = random.randint(search_y_left, search_y_right)
    return [new_bbox_x_c, new_bbox_y_c]


def sample_new_bbox_center(img_shape, bbox_h, bbox_w, safe_restrict=20):
    """
        bbox产生的范围
    :param safe_restrict:
    :param img_shape:
    :param bbox_h:
    :param bbox_w:
    :return:
    """
    # sampling space
    h, w, n_channels = img_shape

    # 检查横纵坐标是否是反的
    search_x_left, search_y_left, search_x_right, search_y_right =  bbox_w/2, bbox_h/2, w - bbox_w/2 - safe_restrict,\
                                                                    h - bbox_h/2 - safe_restrict

    # check this out
    # if x_left <= w / 2:  # ??????????? -> 仅仅是一个搜索范围的问题
    #     search_x_left, search_y_left, search_x_right, search_y_right = w * 0.6, h / 2, w * 0.75, h * 0.75
    # else:
    #     search_x_left, search_y_left, search_x_right, search_y_right = w * 0.25, h / 2, w * 0.5, h * 0.75

    result = [search_x_left, search_y_left, search_x_right, search_y_right]
    result = [int(x) for x in result]

    return result


def img_paths2label_paths(img_paths):
    """get labels' path from images' path"""
    return [img_path.replace('.jpg', '.txt') for img_path in img_paths]


def random_search(all_labels, cropped_label, shape, n_paste=1, iou_thresh=0.2):
    """
        搜索出一个框

    :param all_labels:
    :param cropped_label:
    :param shape:
    :param n_paste:
    :param iou_thresh:
    :return:
    """
    cls, (bbox_h, bbox_w) = cropped_label
    # TODO: 这里会产生一个bug，当bbox等于H-1或者W-1时，后续变换会越界
    center_search_space = sample_new_bbox_center(shape, bbox_h, bbox_w)

    n_success = 0
    n_trials = 0
    new_bboxes = []
    # 当尝试次数大于20次就停止
    while n_success < n_paste and n_trials < 20:
        new_bbox_x_center, new_bbox_y_center = uniform_sample(center_search_space)

        # bug 如果box w为奇数，那么会少一个像素点(fixed)
        new_bbox_x_left, new_bbox_y_left = int(new_bbox_x_center - 0.5 * bbox_w), int(new_bbox_y_center - 0.5 * bbox_h)
        new_bbox_x_right, new_bbox_y_right = new_bbox_x_left + bbox_w, new_bbox_y_left + bbox_h

        new_bbox = [cls, new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right]
        # 计算iou
        ious = [compute_iou(new_bbox, bbox_t) for bbox_t in all_labels]
        if max(ious) > iou_thresh:
            continue
        n_success += 1
        n_trials += 1
        # temp.append(new_bbox)
        new_bboxes.append(new_bbox)

    return new_bboxes

# ---------------------------------------- modify by zx ----------------------------------------------------------------
# modify time: 2020.6.11 11.18
# ----------------------------------------------------------------------------------------------------------------------
from xml_utils import read_label_xml, write_label_xml


def paste_small_objects_to_single_img(img_path, label_path, cropped_images, cropped_dir, save_img_dir, save_anno_dir,
                                      n_bboxes=6, prob=1.0, origin_rescale=False,
                                      origin_rescaled_size=800, cropped_rescale=False,
                                      cropped_rescale_size=150):
    """
            按照一定的概率去paste
    :param cropped_rescale_size:
    :param cropped_rescale:
    :param origin_rescaled_size:
    :param origin_rescale:          是否resize原始尺寸，使得crop图像不至于太小
    :param prob:
    :param save_anno_dir:
    :param save_img_dir:
    :param cropped_dir:
    :param img_path:
    :param label_path:
    :param cropped_images:
    :param n_bboxes:
    :type cropped_images: dict
    :return:
    """
    if random.randint(0, 1) > prob:
        return

    # 检查dir是否存在
    ensure_dir_exists(save_img_dir)
    ensure_dir_exists(save_anno_dir)

    origin_image = cv2_im_read(img_path)
    origin_labels = read_label_xml(label_path)

    # TODO: 调节原始图像的大小，不然融合起来有点奇怪
    if origin_rescale:
        annotations = np.array(origin_labels)
        cls = annotations[:, 0]
        bboxes = annotations[:, 1:]
        bboxes = np.array(bboxes, dtype=np.int16)

        _data = {'image': origin_image, 'bboxes': bboxes, 'bbox_labels': cls}
        h, w, _ = origin_image.shape
        if h > w:
            aug = get_aug([Resize(p=1, height=origin_rescaled_size, width=int(w * (origin_rescaled_size / h)))])
        else:
            aug = get_aug([Resize(p=1, width=origin_rescaled_size, height=int(h * (origin_rescaled_size / w)))])

        _data = aug(**_data)

        # 恢复成原始的label样子
        # bug： 得到的结果是[cls, [bbox]]，故无法与后续进行融合
        # _boxes = _data['bboxes']
        # _boxes = [ [int(_x) for _x in _box] for _box in _boxes]
        # origin_labels = [[_data['bbox_labels'][_idx]] for _idx, x in enumerate(_boxes)]

        origin_image = _data['image']
        origin_labels = list()
        for _idx, _box in enumerate(_data['bboxes']):
            _tmp_list = list()
            _cls = _data['bbox_labels'][_idx]

            _tmp_list.append(_cls)
            _tmp_list.extend([int(_x) for _x in _box])
            origin_labels.append(_tmp_list)


    # delete (测试)
    # cv2_img_show(origin_image)
    # draw_annotation_to_image(origin_image, origin_labels)

    if len(origin_labels) >= n_bboxes:
        return

    # 添加所有的锚框，方便后续写入文件
    all_labels = []
    all_labels.extend(origin_labels)

    # 从待选的crop图像中选取n个填充到原图像中
    n_cropped_images = len(cropped_images.keys())
    list_cropped_images = list(cropped_images.keys())
    tmp_idx = np.random.permutation(n_cropped_images)

    # 往图像中插入图
    for i in range(n_bboxes):
        # 读取crop图像
        cropped_id = list_cropped_images[tmp_idx[i]]
        cropped_img_path = os.path.join(cropped_dir, cropped_id + '.jpg')
        cropped_cls = cropped_images.get(cropped_id)

        roi = cv2_im_read(cropped_img_path)
        _h, _w, _ = roi.shape
        # TODO: resize 小图像
        if cropped_rescale:
            # 足够小
            if max(_h, _w) < 80:
                roi = cv2.resize(roi, (int(_w * (cropped_rescale_size / _h)), cropped_rescale_size),
                                                                                        interpolation=cv2.INTER_CUBIC)

        # debug
        # _h, _w, _ = roi.shape
        # cv2.imshow('', roi)
        # cv2.waitKey(0)

        cropped_label = [cropped_cls, roi.shape[:2]]

        # searching for places
        new_bboxes = random_search(all_labels, cropped_label, origin_image.shape, n_paste=1, iou_thresh=0.2)

        for new_label in new_bboxes:
            all_labels.append(new_label)
            bbox_left, bbox_top, bbox_right, bbox_bottom = new_label[1], new_label[2], new_label[3], new_label[4]
            try:
                # 随机翻转
                roi = random_flip_bbox(roi)
                # TODO: 图像融合，尝试泊松融合
                # fuse_img(origin_image, roi, bbox_left, bbox_top, bbox_right, bbox_bottom, mode='normal')
                fuse_img(origin_image, roi, bbox_left, bbox_top, bbox_right, bbox_bottom, mode='cutmix')
            except ValueError as e:
                print(e)

    # debug
    # draw_annotation_to_image(origin_image, all_labels)

    # 保存结果
    # step 1 命名
    # [source_file_name]_n_bboxes_timestamp.jpg
    # [source_file_name]_n_bboxes_timestamp.xml
    aug_img_file_name = '{}_pasted_{}_boxes_{}.jpg'.format(os.path.basename(img_path).split('.jpg')[0], n_bboxes,
                                              int(time.time() * 1000))
    aug_anno_file_name = aug_img_file_name.replace('.jpg', '.xml')
    aug_img_path = os.path.join(save_img_dir, aug_img_file_name)
    aug_anno_path = os.path.join(save_anno_dir, aug_anno_file_name)

    # step 2 保存图像
    cv_imwrite(aug_img_path, origin_image)

    # step 3 保存label，这里使用xml形式
    write_label_xml(aug_anno_path, all_labels, origin_image.shape)


def cv2_im_read(img_path):
    """
        解决cv2不支持中文路径的问题
    :param img_path:
    :return:
    """
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_img_show(img, title='debug'):
    """
        简化图像显示，用于debug
    :param img:
    :param title:
    :return:
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)


def draw_annotation_to_image(img, annotations):
    """
        用于debug，展示图像bbox
    :param img:
    :param annotations:
    :return:
    """
    for anno in annotations:
        cl, x1, y1, x2, y2 = anno
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cl, (int((x1 + x2) / 2), y1 - 5), font, fontScale=0.8, color=(0, 0, 255))
    cv2_img_show(img)
    # cv2.imwrite(save_img_dir, img)


def fuse_img(origin_image, roi, bbox_left, bbox_top, bbox_right, bbox_bottom, mode='normal', cutmix_lambda=0.5):
    """
            尝试融合两个图像
    :param cutmix_lambda:
    :param mode:                cutmix, normal, poisson
    :param origin_image:
    :param roi:
    :param bbox_left:
    :param bbox_top:
    :param bbox_right:
    :param bbox_bottom:
    :return:
    """
    if mode == 'normal':
        # (bug) could not broadcast input array from shape (93,97,3) into shape (92,97,3)
        # -> fixed, shawn_zhu 2020.6.12 10:31
        origin_image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
    if mode == 'poisson':
        # TODO: 泊松融合
        pass
    if mode == 'cutmix':
        # TODO: cutmix
        _tmp_roi = origin_image[bbox_top:bbox_bottom, bbox_left:bbox_right] * cutmix_lambda + roi * (1-cutmix_lambda)
        origin_image[bbox_top:bbox_bottom, bbox_left:bbox_right] = _tmp_roi

    return origin_image

def cv_imwrite(filename, src):
    """
        解决cv2不支持中文的问题
    :param filename:
    :param src:
    :return:
    """
    cv2.imencode('.jpg',src)[1].tofile(filename)

# 图像resize
from albumentations import (
    BboxParams,
    Resize,
    Compose
)
def get_aug(_aug, min_area=0., min_visibility=0.):
    """
        获得augmentations的resize
    :param _aug:
    :param min_area:
    :param min_visibility:
    :return:
    """
    return Compose(_aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['bbox_labels']))
