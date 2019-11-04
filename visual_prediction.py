from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
import random
import cv2
from detectron2.utils.visualizer import Visualizer
import view_tools

data_cate = "Leucorrhea_val"

def setup_data():
    register_coco_instances("Leucorrhea_train", {}, "./Datasets/RecGrapReslutForNet/train.json",
                            "./Datasets/RecGrapReslutForNet/Images")
    register_coco_instances("Leucorrhea_val", {}, "./Datasets/RecGrapReslutForNet/val.json",
                            "./Datasets/RecGrapReslutForNet/Images")


def setup():
    setup_data()
    cfg = get_cfg()
    cfg.merge_from_file("configs/MyConfig/version0.01.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg


def main():
    cfg = setup()
    predictor = DefaultPredictor(cfg)
    cells_metadata = MetadataCatalog.get(data_cate)
    dataset_dicts = DatasetCatalog.get(data_cate)
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v_pre = Visualizer(im[:, :, ::-1],
                       metadata=cells_metadata,
                       scale=0.5
                       )
        v_pre = v_pre.draw_instance_predictions(outputs["instances"].to("cpu"))
        im_pre = v_pre.get_image()

        v_origin = Visualizer(im[:, :, ::-1], metadata=cells_metadata, scale=0.5)
        #im_origin = im.copy()
        #annotations = d["annotations"]
        #bbox = []
        #labels = []
        #for annotation in annotations:
            #bbox.append(annotation["bbox"])
            #labels.append(annotation["category_id"])
        #view_tools.draw_one_image(im_origin, bbox, labels)
        v_origin = v_origin.draw_dataset_dict(d)
        im_origin = v_origin.get_image()
        view_tools.showImages([im_origin, im_pre], ['img_origin', 'img_predict'])


if __name__ == "__main__":
    main()
