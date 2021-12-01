from torch.jit import annotate

import os
import pdb
import torch
import torchvision

from xml.etree import ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

import pdb

_classes = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

_class_to_ind = dict(zip(_classes, range(len(_classes))))



def parse_xml_12_filter_img(xml_file, re_xml_file, new_re_xml_file):
    tree_gt = ET.parse(xml_file)
    tree_re = ET.parse(re_xml_file)
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = tree_gt.find('folder').text
    ET.SubElement(root, 'filename').text = tree_gt.find('filename').text
    
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = tree_gt.find('source').find('database').text
    ET.SubElement(source, 'annotation').text = tree_gt.find('source').find('annotation').text
    ET.SubElement(source, 'image').text = tree_gt.find('source').find('image').text
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = tree_gt.find('size').find('width').text
    ET.SubElement(size, 'height').text = tree_gt.find('size').find('height').text
    ET.SubElement(size, 'depth').text = tree_gt.find('size').find('depth').text
    
    ET.SubElement(root, 'segmented').text = tree_gt.find('segmented').text
    
    objs = tree_gt.findall('object')
    objs_re = tree_re.findall('object')
    boxes_gt = []
    boxes_re = []
    classes_gt=[]
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        try:
            cls_gt = _class_to_ind[obj.find('name').text.lower().strip()]
        except BaseException:
            print(f'_class_to_ind:{_class_to_ind}')
            print(f"obj.find('name').text.lower().strip(): {obj.find('name').text.lower().strip()}")
            pdb.set_trace()
        classes_gt.append(cls_gt)
        boxes_gt.append([cls_gt, x1, y1, x2, y2])
    boxes_gt = torch.tensor(boxes_gt, dtype=torch.float)
    classes_gt = torch.tensor(classes_gt).unique()
    
    for ix, obj in enumerate(objs_re):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls_re = _class_to_ind[obj.find('name').text.lower().strip()]
        boxes_re.append([cls_re, x1, y1, x2, y2])
                                    
    boxes_re = torch.tensor(boxes_re, dtype=torch.float)

    
    keep_box_filtered = []
    for cls in classes_gt.tolist():
        idx_box_gt = (boxes_gt[:, 0] == cls)
        idx_box_re = (boxes_re[:, 0] == cls)
        boxes_re_keep = boxes_re[idx_box_re]
        ious = torchvision.ops.box_iou(boxes_re_keep[:,1:], boxes_gt[idx_box_gt][:,1:])
        idx_keep = (ious >= 0.55).sum(1) > 0
        keep_box_filtered.append(boxes_re_keep[idx_keep])

    keep_box_filtered = torch.cat(keep_box_filtered, dim=0)
    num_boxes = boxes_re.shape[0]
    num_boxes_keep = keep_box_filtered.shape[0]

    for obj in keep_box_filtered.tolist():
        obj_struct = ET.SubElement(root, 'object')
        ET.SubElement(obj_struct, 'name').text = _classes[int(obj[0])]
        bndbox = ET.SubElement(obj_struct, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj[1])
        ET.SubElement(bndbox, 'ymin').text = str(obj[2])
        ET.SubElement(bndbox, 'xmax').text = str(obj[3])
        ET.SubElement(bndbox, 'ymax').text = str(obj[4])
    xmltsr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=6*' ')
    
    # open(re_xml_file, 'w').close()
    
    os.makedirs(os.path.dirname(new_re_xml_file), exist_ok=True)
    with open(new_re_xml_file, 'w') as f:
        f.write(xmltsr)
    
    return num_boxes, num_boxes_keep



def parse_xml_07_filter_img(xml_file, re_xml_file, new_re_xml_file):
    tree_gt = ET.parse(xml_file)
    tree_re = ET.parse(re_xml_file)
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = tree_gt.find('folder').text
    ET.SubElement(root, 'filename').text = tree_gt.find('filename').text

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = tree_gt.find('source').find('database').text
    ET.SubElement(source, 'annotation').text = tree_gt.find('source').find('annotation').text
    ET.SubElement(source, 'image').text = tree_gt.find('source').find('image').text
    ET.SubElement(source, 'flickrid').text = tree_gt.find('source').find('flickrid').text
    
    owner = ET.SubElement(root, 'owner')
    ET.SubElement(owner, 'flickrid').text = tree_gt.find('owner').find('flickrid').text
    ET.SubElement(owner, 'name').text = tree_gt.find('owner').find('name').text
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = tree_gt.find('size').find('width').text
    ET.SubElement(size, 'height').text = tree_gt.find('size').find('height').text
    ET.SubElement(size, 'depth').text = tree_gt.find('size').find('depth').text
    
    ET.SubElement(root, 'segmented').text = tree_gt.find('segmented').text
    
    objs = tree_gt.findall('object')
    objs_re = tree_re.findall('object')
    boxes_gt = []
    boxes_re = []
    classes_gt=[]
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        try:
            cls_gt = _class_to_ind[obj.find('name').text.lower().strip()]
        except BaseException:
            print(f'_class_to_ind:{_class_to_ind}')
            print(f"obj.find('name').text.lower().strip(): {obj.find('name').text.lower().strip()}")
            pdb.set_trace()
        classes_gt.append(cls_gt)
        boxes_gt.append([cls_gt, x1, y1, x2, y2])
    boxes_gt = torch.tensor(boxes_gt, dtype=torch.float)
    classes_gt = torch.tensor(classes_gt).unique()
    
    for ix, obj in enumerate(objs_re):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls_re = _class_to_ind[obj.find('name').text.lower().strip()]
        boxes_re.append([cls_re, x1, y1, x2, y2])
                                    
    boxes_re = torch.tensor(boxes_re, dtype=torch.float)

    
    keep_box_filtered = []
    for cls in classes_gt.tolist():
        idx_box_gt = (boxes_gt[:, 0] == cls)
        idx_box_re = (boxes_re[:, 0] == cls)
        boxes_re_keep = boxes_re[idx_box_re]
        ious = torchvision.ops.box_iou(boxes_re_keep[:,1:], boxes_gt[idx_box_gt][:,1:])
        idx_keep = (ious >= 0.55).sum(1) > 0
        keep_box_filtered.append(boxes_re_keep[idx_keep])

    keep_box_filtered = torch.cat(keep_box_filtered, dim=0)
    num_boxes = boxes_re.shape[0]
    num_boxes_keep = keep_box_filtered.shape[0]

    for obj in keep_box_filtered.tolist():
        obj_struct = ET.SubElement(root, 'object')
        ET.SubElement(obj_struct, 'name').text = _classes[int(obj[0])]
        bndbox = ET.SubElement(obj_struct, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(obj[1]))
        ET.SubElement(bndbox, 'ymin').text = str(int(obj[2]))
        ET.SubElement(bndbox, 'xmax').text = str(int(obj[3]))
        ET.SubElement(bndbox, 'ymax').text = str(int(obj[4]))
    xmltsr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=6*' ')
    
    # open(re_xml_file, 'w').close()
    
    os.makedirs(os.path.dirname(new_re_xml_file), exist_ok=True)
    with open(new_re_xml_file, 'w') as f:
        f.write(xmltsr)
    
    return num_boxes, num_boxes_keep

if __name__ == "__main__":


    # Dealing VOC2007 labels
    voc_root = '/home/LiaoMingxiang/Workspace/weak_det/D-MIL/D-MIL.pytorch-master/data/VOCdevkit2007/VOC2007'
    root_gt = os.path.join(voc_root, 'Annotations')
    root_re = os.path.join(voc_root, 'Annotations_re')
    root_dir= os.path.join(voc_root, 'Annotations_re_new_055')
    file_list = os.listdir(root_re)
    num_boxes = 0
    num_boxes_keep = 0
    for f in tqdm(file_list):
        xml_file = os.path.join(root_gt, f)
        re_xml_file = os.path.join(root_re, f)
        new_re_xml_file = os.path.join(root_dir, f)
        nb, nb_keep = parse_xml_07_filter_img(xml_file=xml_file, re_xml_file=re_xml_file, new_re_xml_file=new_re_xml_file)
        num_boxes += nb
        num_boxes_keep += nb_keep
    
    print(f'num_boxes of VOC07: {num_boxes}')
    print(f'num_boxes_keep of VOC07: {num_boxes_keep}')
    print(f'keep_ratio of VOC07: {num_boxes_keep / num_boxes}')

    # Dealing VOC2012 labels
    voc_root = '/home/LiaoMingxiang/Workspace/weak_det/D-MIL/D-MIL.pytorch-master/data/VOCdevkit2012/VOC2012'
    root_gt = os.path.join(voc_root, 'Annotations')
    root_re = os.path.join(voc_root, 'Annotations_re')
    root_dir= os.path.join(voc_root, 'Annotations_re_new_055')
    file_list = os.listdir(root_re)
    num_boxes = 0
    num_boxes_keep = 0
    for f in tqdm(file_list):
        xml_file = os.path.join(root_gt, f)
        re_xml_file = os.path.join(root_re, f)
        new_re_xml_file = os.path.join(root_dir, f)
        nb, nb_keep = parse_xml_12_filter_img(xml_file=xml_file, re_xml_file=re_xml_file, new_re_xml_file=new_re_xml_file)
        num_boxes += nb
        num_boxes_keep += nb_keep
    
    print(f'num_boxes of VOC12: {num_boxes}')
    print(f'num_boxes_keep of VOC12: {num_boxes_keep}')
    print(f'keep_ratio of VOC12: {num_boxes_keep / num_boxes}')

    
    # xml_file = '/home/LiaoMingxiang/Workspace/weak_det/D-MIL.pytorch-master/data/VOCdevkit2007/VOC2007/Annotations/000005.xml'
    # re_xml_file = '/home/LiaoMingxiang/Workspace/weak_det/D-MIL.pytorch-master/data/VOCdevkit2007/VOC2007/Annotations_re/000005.xml'
    # new_re_xml_file = '/home/LiaoMingxiang/Workspace/weak_det/D-MIL.pytorch-master/data/VOCdevkit2007/VOC2007/Annotations_re_new/000005.xml'

    # parse_xml_07_filter_img(xml_file=xml_file, re_xml_file=re_xml_file, new_re_xml_file=new_re_xml_file)