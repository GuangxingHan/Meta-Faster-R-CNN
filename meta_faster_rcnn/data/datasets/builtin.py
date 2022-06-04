"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""
import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .builtin_meta_pascal_voc import _get_builtin_metadata_pascal_voc
from .meta_pascal_voc import register_meta_pascal_voc
from detectron2.data import MetadataCatalog

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train_nonvoc": ("coco/trainval2014", "coco/new_annotations/final_split_non_voc_instances_train2014.json"), # by default no_smaller_32
    "coco_2014_train_nonvoc_with_small": ("coco/trainval2014", "coco/new_annotations/final_split_non_voc_instances_train2014_with_small.json"), # includeing all boxes

    "coco_2014_train_voc_10_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_10_shot_instances_train2014.json"),
    "coco_2014_train_voc_1_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_1_shot_instances_train2014.json"),
    "coco_2014_train_voc_2_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_2_shot_instances_train2014.json"),
    "coco_2014_train_voc_3_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_3_shot_instances_train2014.json"),
    "coco_2014_train_voc_5_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_5_shot_instances_train2014.json"),
    "coco_2014_train_voc_30_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_30_shot_instances_train2014.json"),

    "coco_2014_train_full_10_shot": ("coco/trainval2014", "coco/new_annotations/full_class_10_shot_instances_train2014.json"),
    "coco_2014_train_full_1_shot": ("coco/trainval2014", "coco/new_annotations/full_class_1_shot_instances_train2014.json"),
    "coco_2014_train_full_2_shot": ("coco/trainval2014", "coco/new_annotations/full_class_2_shot_instances_train2014.json"),
    "coco_2014_train_full_3_shot": ("coco/trainval2014", "coco/new_annotations/full_class_3_shot_instances_train2014.json"),
    "coco_2014_train_full_5_shot": ("coco/trainval2014", "coco/new_annotations/full_class_5_shot_instances_train2014.json"),
    "coco_2014_train_full_30_shot": ("coco/trainval2014", "coco/new_annotations/full_class_30_shot_instances_train2014.json"),
}

def register_all_coco(root):
    # for prefix in ["novel",]: #"all", 
    for shot in [1, 2, 3, 5, 10, 30]:
        for seed in range(1, 10):
            name = "coco_2014_train_voc_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_COCO["coco"][name] = ("coco/trainval2014", "coco/new_annotations/seed{}/{}_shot_instances_train2014.json".format(seed, shot))

            name = "coco_2014_train_full_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_COCO["coco"][name] = ("coco/trainval2014", "coco/new_annotations/seed{}/full_class_{}_shot_instances_train2014.json".format(seed, shot))


    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = '' if seed == 0 else '_seed{}'.format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed)
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid)
                        keepclasses = "base_novel_{}".format(sid) \
                            if prefix == 'all' else "novel{}".format(sid)
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid))

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(name,
                                 _get_builtin_metadata_pascal_voc("pascal_voc_fewshot"),
                                 os.path.join(root, dirname), split, year,
                                 keepclasses, sid)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
_root = os.getenv("DETECTRON2_DATASETS", "datasets/pascal_voc")
register_all_pascal_voc(_root)
