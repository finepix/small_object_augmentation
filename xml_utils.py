#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/11
# @Author  : shawn_zhu
# @Site    : 
# @File    : xml_utils.py
# @Software: PyCharm

"""
import xml.etree.ElementTree as ET
import xml.dom.minidom


def read_label_xml(xml_label_path):
    """
        读取xml的label

    :param xml_label_path:
    :return:
    """
    tree = ET.parse(xml_label_path)
    root = tree.getroot()

    # 所有的label（cls[0]， bbox[1:4]）
    rs_labels = list()
    for obj in root.iter('object'):
        _label = list()
        name = obj.find('name').text
        _label.append(name)

        xml_bbox = obj.find('bndbox')
        box = [int(xml_bbox.find('xmin').text), int(xml_bbox.find('ymin').text),
               int(xml_bbox.find('xmax').text), int(xml_bbox.find('ymax').text)]

        _label.extend(box)
        rs_labels.append(_label)

    return rs_labels

def create_element_ndoe(doc, tag, value):
    """
        添加xmlnode
    :param doc:
    :param tag:
    :param value:
    :return:
    """
    node = doc.createElement(tag)
    text_node = doc.createTextNode(value)

    node.appendChild(text_node)
    return node

def create_child_node(doc, tag, value, parent_node):
    """
            创建child节点
    :param doc:
    :param tag:
    :param value:
    :param parent_node:
    :return:
    """
    child_node = create_element_ndoe(doc, tag, value)
    parent_node.appendChild(child_node)


def create_object_node(doc, label):
    """

        label: [cls, bbox]
    :param doc:
    :param label:
    :return:
    """
    name, box = label[0], label[1:]

    object_node = doc.createElement('object')

    create_child_node(doc, 'name', name, object_node)

    bndbox_node = doc.createElement('bndbox')
    create_child_node(doc, 'xmin', str(box[0]), bndbox_node)
    create_child_node(doc, 'ymin', str(box[1]), bndbox_node)
    create_child_node(doc, 'xmax', str(box[2]), bndbox_node)
    create_child_node(doc, 'ymax', str(box[3]), bndbox_node)
    object_node.appendChild(bndbox_node)

    return object_node


def create_size_node(doc, size):
    """
        创建size node
    :param doc:
    :param size:
    :return:
    """
    h, w, c = size
    size_node = doc.createElement('size')

    create_child_node(doc, 'width', str(w), size_node)
    create_child_node(doc, 'height', str(h), size_node)
    create_child_node(doc, 'depth', str(c), size_node)

    return size_node


def write_xml_file(doc, file_name, _tmp_file_name='.tmp/_tmp.xml', _encoding='utf-8'):
    """
        写入xml文件
    :param _encoding:
    :param doc:
    :param file_name:
    :param _tmp_file_name:
    :return:
    """
    with open(_tmp_file_name, 'w', encoding=_encoding) as _tmp_file:
        doc.writexml(_tmp_file, addindent=' ' * 4, newl='\n', encoding='utf-8')

    # 删除第一行标记
    with open(_tmp_file_name, encoding=_encoding) as f_in:
        lines = f_in.readlines()
        with open(file_name, 'w', encoding=_encoding) as f_out:
            for line in lines[1:]:
                if line.strip():
                    f_out.write(line)

    # print(f'write file:{file_name} finished.')




def write_label_xml(anno_path, labels, origin_shape, root_node_name='annotation'):
    """
        labels: [[cls, x1, y1, x2, y2]]
    :param root_node_name:
    :param origin_shape:
    :param anno_path:
    :param labels:
    :return:
    """
    _dom = xml.dom.getDOMImplementation()
    doc = _dom.createDocument(None, root_node_name, None)
    root_node = doc.documentElement

    # 添加size node
    root_node.appendChild(create_size_node(doc, origin_shape))

    # 添加obj node
    for label in labels:
        obj_node = create_object_node(doc, label)
        root_node.appendChild(obj_node)

    write_xml_file(doc, anno_path)


