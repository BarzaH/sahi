# OBSS SAHI Tool
# Code written by Fatih C Akyon and Kadir Nar, 2021.

import logging
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch import Tensor, nn

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from sahi.utils.torch import to_float_tensor

logger = logging.getLogger(__name__)


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class RTDetrPyTorchDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["torch", "torchvision", "rtdetr"])

    def load_model(self):
        from rtdetr.core import YAMLConfig

        # read config params
        assert self.config_path is not None
        self.whole_rtdetr = YAMLConfig(self.config_path, resume=self.model_path)
        self.set_model()

    def set_model(self, *_, **__):
        """
        Sets the underlying TorchVision model.
        """
        check_requirements(["torch", "torchvision"])
        self.model = self.whole_rtdetr.model
        state = torch.load(self.model_path, map_location="cpu")
        if getattr(self, "model", None) and "model" in state:
            if is_parallel(self.model):
                self.model.module.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # set category_mapping
        if self.category_mapping is None:
            self.category_mapping = {
                "0": "Саженец (high)",
                "1": "Хвоя (low)",
            }

    def load_state_dict(self, path):
        """load state dict"""
        state = torch.load(path, map_location=self.device)
        if getattr(self, "model", None) and "model" in state:
            if is_parallel(self.model):
                self.model.module.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state["model"])

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model
        the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
                3 channel image should be in RGB order.
        """

        # arrange model input size
        orig_target_sizes = image.shape[:2]
        w, h = orig_target_sizes
        if w != 640 or h != 640:
            image = T.Resize(640, max_size=640)
        image = to_float_tensor(image)
        image = image.to(self.device)
        prediction_result = self.model(image)
        # pylint: disable=not-callable
        prediction_result = self.whole_rtdetr.postprocessor(
            prediction_result, Tensor([orig_target_sizes]).to(self.device)
        )
        # pylint: enable=not-callable
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        for image_predictions in original_predictions:
            object_prediction_list_per_image = []

            # get indices of boxes with score > confidence_threshold
            scores = image_predictions["scores"].cpu().detach().numpy()
            selected_indices = np.where(scores > self.confidence_threshold)[0]

            # parse boxes, masks, scores, category_ids from predictions
            category_ids = list(image_predictions["labels"][selected_indices].cpu().detach().numpy())
            boxes = list(image_predictions["boxes"][selected_indices].cpu().detach().numpy())
            scores = scores[selected_indices]

            # check if predictions contain mask
            masks = image_predictions.get("masks", None)
            if masks is not None:
                masks = list(image_predictions["masks"][selected_indices].cpu().detach().numpy())
            else:
                masks = None

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = shift_amount_list[0]
            full_shape = None if full_shape_list is None else full_shape_list[0]

            for ind, box in enumerate(boxes):
                if masks is not None:
                    mask = get_coco_segmentation_from_bool_mask(np.array(masks[ind]))
                else:
                    mask = None

                object_prediction = ObjectPrediction(
                    bbox=box,
                    segmentation=mask,
                    category_id=int(category_ids[ind]),
                    category_name=self.category_mapping[str(int(category_ids[ind]))],
                    shift_amount=shift_amount,
                    score=scores[ind],
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
