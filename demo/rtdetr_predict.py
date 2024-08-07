import torch

from sahi.predict import predict

with torch.no_grad():
    predict(
        model_type="pytorch_rtdetr",
        model_path="/home/sema/RT-DETR/rtdetr_pytorch/output/stage0/rtdetr_r50vd_6x_coco/checkpoint0057.pth",
        model_config_path="/home/sema/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco_singleton.yml",
        model_confidence_threshold=0.4,
        force_postprocess_type=True,
        model_device="cuda:0",
        model_category_mapping={"0": "Саженец (high)", "1": "Хвоя (low)"},
        source="/mnt/nvstorage/gis/seedling/data/tiled_coco/2560/images/test/",
        dataset_json_path="/mnt/nvstorage/gis/seedling/data/tiled_coco/2560/anno/test.json",
        # dataset_json_path="/mnt/nvstorage/gis/seedling/data/tiled_coco/640/images/test/test_coco.json",
        no_standard_prediction=True,
        image_size=640,
        slice_height=640,
        slice_width=640,
        postprocess_match_threshold=0.25,
        visual_hide_labels=True,
        return_dict=True,
        verbose=0,
    )
