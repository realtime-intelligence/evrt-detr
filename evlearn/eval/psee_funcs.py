import numpy as np
from psee_adt.io.box_loading import BBOX_DTYPE

def convert_prediction_postproc_yolox_to_psee(
    yolox_preds_list, psee_labels_list
):
    # yolox_preds : List[Tensor(x0, y0, x1, y1, obj_conf, cls_conf, cls_pred)]
    # Source: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py

    # psee_labels : List[ndarray[
    #   't','x','y','w','h','class_id','track_id','class_confidence'
    # ]]

    result = []

    for (yolox_preds, psee_labels) in zip(yolox_preds_list, psee_labels_list):
        n = 0

        if yolox_preds is not None:
            n = yolox_preds.shape[0]

        psee_preds = np.zeros((n, ), dtype = BBOX_DTYPE)

        if n == 0:
            result.append(psee_preds)
            continue

        yolox_preds = yolox_preds.detach().cpu().numpy()

        times = np.unique(psee_labels['t'])
        assert len(times) == 1
        time = times[0]

        psee_preds['t'] = time
        psee_preds['x'] = yolox_preds[:, 0]
        psee_preds['y'] = yolox_preds[:, 1]
        psee_preds['w'] = (yolox_preds[:, 2] - yolox_preds[:, 0])
        psee_preds['h'] = (yolox_preds[:, 3] - yolox_preds[:, 1])

        psee_preds['class_id']         = yolox_preds[:, 6]
        psee_preds['class_confidence'] = yolox_preds[:, 5]

        result.append(psee_preds)

    return result

def convert_prediction_postproc_rtdetr_to_psee(
    rtdetr_preds_list, psee_labels_list
):
    # preds_rtdetr_list : List[{ 'boxes', 'labels', 'scores' })
    #   where 'boxes'   : (x0, y0. x1, y1)
    #
    # psee_labels_list : List[ndarray[
    #   't','x','y','w','h','class_id','track_id','class_confidence'
    # ]]
    result = []

    for (rtdetr_preds, psee_labels) in zip(
        rtdetr_preds_list, psee_labels_list
    ):
        n = len(rtdetr_preds['labels'])

        rtdetr_labels = rtdetr_preds['labels'].detach().cpu().float().numpy()
        rtdetr_boxes  = rtdetr_preds['boxes'] .detach().cpu().float().numpy()
        rtdetr_scores = rtdetr_preds['scores'].detach().cpu().float().numpy()

        psee_preds = np.zeros((n, ), dtype = BBOX_DTYPE)

        if n == 0:
            result.append(psee_preds)
            continue

        times = np.unique(psee_labels['t'])
        assert len(times) == 1
        time = times[0]

        psee_preds['t'] = time
        psee_preds['x'] = rtdetr_boxes[:, 0]
        psee_preds['y'] = rtdetr_boxes[:, 1]
        psee_preds['w'] = (rtdetr_boxes[:, 2] - rtdetr_boxes[:, 0])
        psee_preds['h'] = (rtdetr_boxes[:, 3] - rtdetr_boxes[:, 1])

        psee_preds['class_id']         = rtdetr_labels
        psee_preds['class_confidence'] = rtdetr_scores

        result.append(psee_preds)

    return result

