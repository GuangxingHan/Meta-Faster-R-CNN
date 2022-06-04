python update_params_resnet101_fpn.py output/fsod/meta_training_pascalvoc_split2_resnet101_stage_3/model_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 faster_rcnn_train_net.py --num-gpus 4 --dist-url auto \
	--config-file configs/fsod/faster_rcnn_with_fpn_pascalvoc_split2_base_classes_branch.yaml 2>&1 | tee log/faster_rcnn_with_fpn_pascalvoc_split2_base_classes_branch.txt
