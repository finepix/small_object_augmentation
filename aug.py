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


def find_str(filename):
    """????"""
    return dirname(filename[filename.find('train' if 'train' in filename else 'val'):])


def convert(size, box):
    """size , box -> x, y, w, h"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x *= dw
    w *= dw
    y *= dh
    h *= dh
    return x, y, w, h


def convert_boxes(shape, anno_infos, label_txt_dir):
    height, width, _ = shape
    label_file = open(label_txt_dir, 'w')
    for target_id, x1, y1, x2, y2 in anno_infos:
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)
        label_file.write(str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def is_small_object(bbox, thresh):
    """check if the given bbox is small object, iff area <= thresh"""
    return bbox[0] * bbox[1] <= thresh


# def load_txt_label(label_txt_path):
#     return np.loadtxt(label_txt_path, dtype=str)
#
#
# def load_txt_labels(label_dir):
#     return [load_txt_label(label) for label in label_dir]


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

    # TODO: debug
    # assert cls1 == cls2, "2 boxes to compute iou should be same class"

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


def sample_new_bbox_center(img_shape, bbox_h, bbox_w):
    """
        bbox产生的范围
    :param img_shape:
    :param bbox_h:
    :param bbox_w:
    :return:
    """
    # sampling space
    h, w, n_channels = img_shape

    # TODO: 检查横纵坐标是否是反的
    search_x_left, search_y_left, search_x_right, search_y_right =  bbox_w/2, bbox_h/2, w - bbox_w/2, h - bbox_h/2

    # TODO: check this out
    # if x_left <= w / 2:  # ???????????
    #     search_x_left, search_y_left, search_x_right, search_y_right = w * 0.6, h / 2, w * 0.75, h * 0.75
    # else:
    #     search_x_left, search_y_left, search_x_right, search_y_right = w * 0.25, h / 2, w * 0.5, h * 0.75

    result = [search_x_left, search_y_left, search_x_right, search_y_right]
    result = [int(x) for x in result]

    return result


def img_paths2label_paths(img_paths):
    """get labels' path from images' path"""
    return [img_path.replace('.jpg', '.txt') for img_path in img_paths]


def random_search(all_labels, croped_label, shape, n_paste=1, iou_thresh=0.2):
    """
        搜索出一个框

    :param all_labels:
    :param croped_label:
    :param shape:
    :param n_paste:
    :param iou_thresh:
    :return:
    """
    cls, (bbox_h, bbox_w) = croped_label
    center_search_space = sample_new_bbox_center(shape, bbox_h, bbox_w)

    n_success = 0
    n_trials = 0
    new_bboxes = []
    # 当尝试次数大于20次就停止
    while n_success < n_paste and n_trials < 20:
        new_bbox_x_center, new_bbox_y_center = uniform_sample(center_search_space)
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = int(
            new_bbox_x_center - 0.5 * bbox_w), int(new_bbox_y_center - 0.5 * bbox_h), int(
            new_bbox_x_center + 0.5 * bbox_w), int(new_bbox_y_center + 0.5 * bbox_h)
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


def paste_small_objects_to_single_img(img_path, label_path, croped_images, croped_dir, save_img_dir, save_anno_dir,
                                                                        n_bboxes=6):
    """

    :param save_anno_dir:
    :param save_img_dir:
    :param croped_dir:
    :param img_path:
    :param label_path:
    :param croped_images:
    :param n_bboxes:
    :type croped_images: dict
    :return:
    """

    origin_image = cv2_im_read(img_path)
    origin_labels = read_label_xml(label_path)

    # TODO: delete (测试)
    # cv2_img_show(origin_image)
    # draw_annotation_to_image(origin_image, origin_labels)

    if len(origin_labels) >= n_bboxes:
        return

    # TODO: 添加所有的锚框，方便后续写入文件
    all_labels = []
    all_labels.extend(origin_labels)

    # 从待选的crop图像中选取n个填充到原图像中
    n_croped_images = len(croped_images.keys())
    list_croped_images = list(croped_images.keys())
    tmp_idx = np.random.permutation(n_croped_images)

    # 往图像中插入图
    for i in range(n_bboxes):
        # 读取crop图像
        croped_id = list_croped_images[tmp_idx[i]]
        croped_img_path = os.path.join(croped_dir, croped_id + '.jpg')
        croped_cls = croped_images.get(croped_id)

        roi = cv2_im_read(croped_img_path)
        croped_label = [croped_cls, roi.shape[:2]]

        # searching for places
        new_bboxes = random_search(all_labels, croped_label, origin_image.shape, n_paste=1, iou_thresh=0.2)

        for new_label in new_bboxes:
            all_labels.append(new_label)
            bbox_left, bbox_top, bbox_right, bbox_bottom = new_label[1], new_label[2], new_label[3], new_label[4]
            try:
                # 随机翻转
                roi = random_flip_bbox(roi)
                # TODO: 图像融合，尝试泊松融合
                fuse_img(origin_image, roi, bbox_left, bbox_top, bbox_right, bbox_bottom)
            except ValueError as e:
                print(e)

    # TODO: 保存结果

    # TODO: step 1 命名
    # [source_file_name]_n_bboxes_timestamp.jpg
    # [source_file_name]_n_bboxes_timestamp.xml
    aug_img_file_name = '{}_pasted_{}_boxes_{}.jpg'.format(os.path.basename(img_path).split('.jpg')[0], n_bboxes,
                                              int(time.time() * 1000))
    aug_anno_file_name = aug_img_file_name.replace('.jpg', '.xml')
    aug_img_path = os.path.join(save_img_dir, aug_img_file_name)
    aug_anno_path = os.path.join(save_anno_dir, aug_anno_file_name)

    # TODO: step 2 保存图像
    cv_imwrite(aug_img_path, origin_image)

    # TODO: step 3 保存label，这里使用xml形式
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

def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    """
        保存crop的图像，从
    :param save_crop_base_dir:
    :param image_dir:
    :param idx:
    :param roi:
    :return:
    """
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    ensure_dir_exists(crop_save_dir)
    crop_img_save_dir = join(crop_save_dir, basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def fuse_img(origin_image, roi, bbox_left, bbox_top, bbox_right, bbox_bottom):
    """
            尝试融合两个图像
    :param origin_image:
    :param roi:
    :param bbox_left:
    :param bbox_top:
    :param bbox_right:
    :param bbox_bottom:
    :return:
    """
    # TODO: (bug) could not broadcast input array from shape (93,97,3) into shape (92,97,3)
    origin_image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi

    return origin_image

def cv_imwrite(filename, src):
    """
        解决cv2不支持中文的问题
    :param filename:
    :param src:
    :return:
    """
    cv2.imencode('.jpg',src)[1].tofile(filename)