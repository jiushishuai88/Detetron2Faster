from detectron2.data import DatasetCatalog, MetadataCatalog
from Datasets.LeucorrheaDataset import get_dicts
from detectron2.engine import DefaultTrainer, default_argument_parser, launch, default_setup
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.checkpoint import DetectionCheckpointer
import os
import detectron2.utils.comm as comm
from detectron2.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup_data():
    str = ["a", "b", "c", "d", "e", "f", "g"]
    # DatasetCatalog.register("Leucorrhea_train", lambda: get_dicts(train=True))
    # MetadataCatalog.get("Leucorrhea_train").set(thing_classes=str)
    # DatasetCatalog.register("Leucorrhea_val", lambda: get_dicts(train=False))
    # MetadataCatalog.get("Leucorrhea_val").set(thing_classes=str)
    register_coco_instances("Leucorrhea_train", {}, "./Datasets/RecGrapReslutForNet/train.json",
                            "./Datasets/RecGrapReslutForNet/Images")
    register_coco_instances("Leucorrhea_val", {}, "./Datasets/RecGrapReslutForNet/val.json",
                            "./Datasets/RecGrapReslutForNet/Images")
    register_coco_instances("Leucorrhea_test", {}, "./Datasets/RecGrapReslutForNet/test.json",
                            "./Datasets/RecGrapReslutForNet/Images")


def setup(args):
    setup_data()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(cfg.OUTPUT_DIR, "model_final.pth"), resume=args.resume
        )

        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
