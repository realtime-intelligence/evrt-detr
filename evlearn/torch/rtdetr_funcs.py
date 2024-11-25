import torch
import torchvision

from torchvision import tv_tensors

def convert_labels_torchvision_to_rtdetr(labels, device, image_size):
    # result = Dict{
    #    'boxes'     : (center_x, center_y, w, h),
    #    'labels'    : class_id,
    #    'orig_size' : (W, H),
    # }
    #
    # where boxes are normalized by the image size to (0, 1).
    #

    result = []
    img_height, img_width = image_size

    for x in labels:
        boxes = x['boxes'].to(device)
        boxes = torchvision.ops.box_convert(
            boxes, in_fmt = boxes.format.value.lower(), out_fmt = 'cxcywh'
        )

        boxes[:, 0] /= img_width
        boxes[:, 2] /= img_width
        boxes[:, 1] /= img_height
        boxes[:, 3] /= img_height

        orig_sizes = torch.as_tensor([ img_width, img_height, ]).to(device)

        result.append({
            'labels'    : x['labels'].long().to(device),
            'boxes'     : boxes,
            'orig_size' : orig_sizes,
        })

    return result

def convert_prediction_postproc_rtdetr_to_torchvision(
    rtdetr_preds_list, image_size
):
    # preds_rtdetr : List[{ 'boxes', 'labels', 'scores' })
    # 'boxes' : (x0, y0. x1, y1)
    result = []

    for rtdetr_preds in rtdetr_preds_list:
        boxes = tv_tensors.BoundingBoxes(
            rtdetr_preds['boxes'],
            format      = 'xyxy',
            canvas_size = image_size,
        )

        labels = rtdetr_preds['labels']
        scores = rtdetr_preds['scores']

        preds = {
            'boxes'  : boxes,
            'labels' : labels,
            'scores' : scores,
        }

        result.append(preds)

    return result

