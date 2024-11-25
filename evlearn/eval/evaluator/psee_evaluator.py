import numpy as np

from psee_adt.io.box_filtering  import filter_boxes
from psee_adt.io.box_loading    import reformat_boxes
from psee_adt.metrics.coco_eval import evaluate_detection

from evlearn.eval.psee_funcs import (
    convert_prediction_postproc_rtdetr_to_psee,
    convert_prediction_postproc_yolox_to_psee
)

COCO_METRICS = [ 'AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L' ]

CAMERA_SETTINGS = {
    'gen1' : {
        'image_height' : 240,
        'image_width'  : 304,
        'min_box_diag' : 30,
        'min_box_side' : 10,
        'skip_from_start_us' : 5e5,
    },
    'gen4' : {
        'image_height' : 720,
        'image_width'  : 1280,
        'min_box_diag' : 60,
        'min_box_side' : 20,
        'skip_from_start_us' : 5e5,
    },
}

# cf. psee_adt : fn evlauate_folders
def run_psee_evaluation(
    preds_list, labels_list, camera, classes, image_size, downsampling_factor
):
    # pylint: disable=too-many-arguments
    assert camera in CAMERA_SETTINGS

    min_box_diag       = CAMERA_SETTINGS[camera]['min_box_diag']
    min_box_side       = CAMERA_SETTINGS[camera]['min_box_side']
    skip_from_start_us = CAMERA_SETTINGS[camera]['skip_from_start_us']

    if downsampling_factor is not None:
        min_box_diag /= downsampling_factor
        min_box_side /= downsampling_factor

        # copy to avoid modifying the original reference
        labels_list = [ np.copy(labels) for labels in labels_list ]

        for labels in labels_list:
            labels['x'] /= downsampling_factor
            labels['y'] /= downsampling_factor
            labels['w'] /= downsampling_factor
            labels['h'] /= downsampling_factor

    def filter_boxes_fn(box):
        return filter_boxes(
            box,
            skip_ts      = skip_from_start_us,
            min_box_diag = min_box_diag,
            min_box_side = min_box_side
        )

    labels_list = [ filter_boxes_fn(box) for box in labels_list ]
    preds_list  = [ filter_boxes_fn(box) for box in preds_list ]

    n_det_objects = sum(len(x) for x in preds_list)

    if n_det_objects == 0:
        return { k : 0 for k in COCO_METRICS }

    result = evaluate_detection(
        labels_list, preds_list,
        height = image_size[0], width = image_size[1], classes = classes,
    )

    return {
        k : result.stats[idx] for (idx, k) in enumerate(COCO_METRICS)
    }

class PseeEvaluator:

    def __init__(
        self, image_size, camera, classes,
        downsampling_factor = 1,
        labels_name         = 'psee_labels',
    ):
        # pylint: disable=too-many-arguments
        self._camera      = camera
        self._down_factor = downsampling_factor
        self._labels_name = labels_name
        self._classes     = classes
        self._image_size  = image_size
        self._down_factor = downsampling_factor

        self._preds  = []
        self._labels = []

    def reset(self):
        self._preds  = []
        self._labels = []

    def append(self, preds, labels, preds_format):
        labels_psee = [ reformat_boxes(x[self._labels_name]) for x in labels ]

        if preds_format == 'rtdetr':
            preds_psee = convert_prediction_postproc_rtdetr_to_psee(
                preds, labels_psee
            )
        elif preds_format == 'yolox':
            preds_psee = convert_prediction_postproc_yolox_to_psee(
                preds, labels_psee
            )
        else:
            raise ValueError(
                f"Unsupported predictions format: '{preds_format}'"
            )

        assert len(preds_psee) == len(labels_psee)

        self._preds  += preds_psee
        self._labels += labels_psee

    def __len__(self):
        return len(self._preds)

    def evaluate(self, reset = True):
        if len(self) == 0:
            return {}

        metrics = run_psee_evaluation(
            preds_list  = self._preds,
            labels_list = self._labels,
            camera      = self._camera,
            classes     = self._classes,
            image_size  = self._image_size,
            downsampling_factor = self._down_factor,
        )

        if reset:
            self.reset()

        return metrics

