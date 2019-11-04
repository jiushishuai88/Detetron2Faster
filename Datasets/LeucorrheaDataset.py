import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import yaml
import cv2


def get_data_path():
    data_path = 'Datasets/RecGrapReslutForNet'
    cwd = os.getcwd()
    path = os.path.join(cwd, data_path)
    return path


def get_bbox_and_labels_from_ini_config(path,id=0):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        objs = []
        for key in config:
            value = config[key]
            area = (value['x2']-value['x1'])*( value['y2']-value['y1'])
            obj = {
                "bbox": [value['x1'], value['y1'], value['x2'], value['y2']],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": value['label'],
                "iscrowd": 0,
                "id": id,
                "area":area
            }
            objs.append(obj)
        return objs


def get_dicts(train=True):
    root = get_data_path()
    imgs = list(sorted(os.listdir(os.path.join(root, 'Images'))))
    anotations = list(sorted(os.listdir(os.path.join(root, "Anotations"))))
    length = len(imgs)
    if train:
        imgs = imgs[0:-2000]
        anotations = anotations[0:-2000]
    else:
        imgs = imgs[-2000:]
        anotations = anotations[-2000:]

    dataset_dicts = []
    for i in range(len(imgs)):
        if train:
            id = i
        else:
            id = length - i
        record = {}
        img_name = imgs[i]
        img_path = os.path.join(root, "Images", img_name)
        anotation_name = anotations[i]
        anotation_path = os.path.join(root, "Anotations", anotation_name)
        height, width = cv2.imread(img_path).shape[:2]

        record["file_name"] = img_path
        record["height"] = height
        record["width"] = width
        objs = get_bbox_and_labels_from_ini_config(anotation_path)
        record["annotations"] = objs
        record["image_id"] = id
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    get_dicts(train=True)


if __name__ == "__main__":
    main()
