import torch

import torchvision
from torchvision import tv_tensors

def convert_labels_torchvision_to_yolox(labels, device, dtype, _image_size):
    # pylint: disable=too-many-locals
    #
    # result : [ N, max_objects, class, xc, yc, w, h ]
    #
    # NOTE: documentation claims that (xc, yc, w, h) should be normalized
    #   to (0, 1).
    #   The training does not work when these quantities are normalized.
    #   Cf. also https://github.com/Megvii-BaseDetection/YOLOX/issues/1654
    #

    batch_size = len(labels)
    max_labels = max(len(x['labels']) for x in labels)

    result = torch.zeros(
        (batch_size, max_labels, 5), device = device, dtype = dtype
    )

    for bidx, label in enumerate(labels):
        n_obj = len(label['labels'])

        cls   = label['labels']
        boxes = label['boxes']

        boxes = torchvision.ops.box_convert(
            boxes, in_fmt = boxes.format.value.lower(), out_fmt = 'cxcywh'
        )

        result[bidx, :n_obj, 0] = cls
        result[bidx, :n_obj, 1] = boxes[:, 0]
        result[bidx, :n_obj, 2] = boxes[:, 1]
        result[bidx, :n_obj, 3] = boxes[:, 2]
        result[bidx, :n_obj, 4] = boxes[:, 3]

    return result

def convert_predictions_postproc_yolox_to_torchvision(
    yolox_preds_list, image_size
):
    # yolox_preds_list : List[
    #   Tensor([ N, (x0, y0, x1, y1, obj_conf, cls_conf, cls_pred) ])
    # ]

    result = [ ]

    for yolox_preds in yolox_preds_list:
        if yolox_preds is None:
            result.append(None)
            continue

        boxes = tv_tensors.BoundingBoxes(
            yolox_preds[:, :4], format = 'xyxy', canvas_size = image_size
        )
        labels = yolox_preds[:, 6]
        scores = yolox_preds[:, 4] * yolox_preds[:, 5]

        result.append({
            'boxes'    : boxes,
            'labels'   : labels,
            'scores'   : scores,
            'obj_conf' : yolox_preds[:, 4],
            'cls_conf' : yolox_preds[:, 5],
        })

    return result

