"""
Defines some tools to handle events.
In particular :
    -> defines events' types
    -> defines functions to read events from binary .dat files using numpy
    -> defines functions to write events to binary .dat files using numpy

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function

from typing import List, Optional, Tuple

import numpy as np
import torch as th
import ipdb
import copy

from data.genx_utils.labels import ObjectLabels

BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'ignored_split', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})

# BBOX_DTYPE2 = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id'],
#                        'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4'],
#                        'offsets': [0, 8, 12, 16, 20, 24, 28], 'itemsize': 40})

BBOX_DTYPE_PRED = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})

YOLOX_PRED_PROCESSED = List[Optional[th.Tensor]]
LOADED_LABELS = List[ObjectLabels]


def reformat_boxes(boxes):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if 't' not in boxes.dtype.names or 'class_confidence' not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=BBOX_DTYPE)
        for name in boxes.dtype.names:
            if name == 'ts':
                new['t'] = boxes[name]
            elif name == 'confidence':
                new['class_confidence'] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes


def loaded_label_to_prophesee(loaded_labels: ObjectLabels) -> np.ndarray:
    loaded_labels.numpy_()
    loaded_label_proph = np.zeros((len(loaded_labels),), dtype=BBOX_DTYPE)
    for name in BBOX_DTYPE.names:
        if name == 'ignored_split':
            # We don't have that and don't need it
            continue
        loaded_label_proph[name] = np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name])
    return loaded_label_proph

def to_prophesee(loaded_label_list: LOADED_LABELS, yolox_pred_list: YOLOX_PRED_PROCESSED, keep_classes: List = []) -> \
        Tuple[List[np.ndarray], List[np.ndarray]]:
    
    assert len(loaded_label_list) == len(yolox_pred_list)
    loaded_label_list_proph = []
    yolox_pred_list_proph = []

    for loaded_labels, yolox_preds in zip(loaded_label_list, yolox_pred_list):
        # TODO: use loaded_label_to_prophesee func here
        time = None
        # --- LOADED LABELS ---

        loaded_labels.numpy_()
        loaded_label_proph = np.zeros((len(loaded_labels),), dtype=BBOX_DTYPE)
        for name in BBOX_DTYPE.names:
            if name == 'ignored_split':
                label = np.asarray(loaded_labels.get('class_id'), dtype=BBOX_DTYPE['class_id'])
                loaded_label_proph[name] = np.isin(label, np.array(keep_classes)).astype(dtype=BBOX_DTYPE[name])
                loaded_label_proph[name] = np.where(loaded_label_proph[name] == 0, 1, 0)
                continue
            # if name == 'class_Id':
            #     loaded_label_proph[name] = np.zeros_like(np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name]))
            loaded_label_proph[name] = np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name])
            # if name =='class_id':
            #     loaded_label_proph[name] = np.asarray(np.zeros_like(loaded_labels.get(name)), dtype=BBOX_DTYPE[name])
            if name == 't':
                time = np.unique(loaded_labels.get(name))
                assert time.size == 1
                time = time.item()

        #modified: we assign the class in keep_classes to 0
        # loaded_label_proph = np.array([(item[0], item[1], item[2], item[3], item[4], 0, item[6], item[7]) 
        #                       for item in loaded_label_proph if int(item[5]) in keep_classes],dtype=BBOX_DTYPE)
        
        loaded_label_list_proph.append(loaded_label_proph)

        # --- YOLOX PREDICTIONS ---
        # Assumes batch of post-processed predictions from YoloX Head.
        # See postprocessing: https://github.com/Megvii-BaseDetection/YOLOX/blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/utils/boxes.py#L32
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        num_pred = 0 if yolox_preds is None else yolox_preds.shape[0]
        yolox_pred_proph = np.zeros((num_pred,), dtype=BBOX_DTYPE)
        if num_pred > 0:
            if isinstance(loaded_labels, np.ndarray):
                yolox_preds = yolox_preds
            else:
                yolox_preds = yolox_preds.detach().cpu().numpy()
            assert yolox_preds.shape == (num_pred, 6)
            yolox_pred_proph['t'] = np.ones((num_pred,), dtype=BBOX_DTYPE['t']) * time
            yolox_pred_proph['x'] = np.asarray(yolox_preds[:, 0], dtype=BBOX_DTYPE['x'])
            yolox_pred_proph['y'] = np.asarray(yolox_preds[:, 1], dtype=BBOX_DTYPE['y'])
            yolox_pred_proph['w'] = np.asarray(yolox_preds[:, 2] - yolox_preds[:, 0], dtype=BBOX_DTYPE['w'])
            yolox_pred_proph['h'] = np.asarray(yolox_preds[:, 3] - yolox_preds[:, 1], dtype=BBOX_DTYPE['h'])
            yolox_pred_proph['class_id'] = np.asarray(yolox_preds[:, 5], dtype=BBOX_DTYPE['class_id'])
            yolox_pred_proph['class_confidence'] = np.asarray(yolox_preds[:, 4], dtype=BBOX_DTYPE['class_confidence'])
        yolox_pred_list_proph.append(yolox_pred_proph)

    return loaded_label_list_proph, yolox_pred_list_proph

def to_prophesee2(loaded_label_list: LOADED_LABELS, yolox_pred_list: YOLOX_PRED_PROCESSED, keep_classes: List = []) -> \
        Tuple[List[np.ndarray], List[np.ndarray]]:
    assert len(loaded_label_list) == len(yolox_pred_list)

    loaded_label_list_proph = []
    yolox_pred_list_proph = []
    for loaded_labels, yolox_preds in zip(loaded_label_list, yolox_pred_list):
        # TODO: use loaded_label_to_prophesee func here
        time = None
        # --- LOADED LABELS ---
        loaded_labels.numpy_()
        loaded_label_proph = np.zeros((len(loaded_labels),), dtype=BBOX_DTYPE)
        for name in BBOX_DTYPE.names:
            if name == 'ignored_split':
                label = np.asarray(loaded_labels.get('class_id'), dtype=BBOX_DTYPE['class_id'])
                loaded_label_proph[name] = np.isin(label, np.array(keep_classes)).astype(dtype=BBOX_DTYPE[name])
                loaded_label_proph[name] = np.where(loaded_label_proph[name] == 0, 1, 0)
                continue
            if name == 'class_id':  #set the car label to 1, pedestrian to 2
                loaded_label_proph[name] = np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name])
                mask_1 = (loaded_label_proph[name] == 1)
                mask_2 = (loaded_label_proph[name] == 2)
                loaded_label_proph[name][mask_1] = 2
                loaded_label_proph[name][mask_2] = 1
                continue
            loaded_label_proph[name] = np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name])
            if name == 't':
                time = np.unique(loaded_labels.get(name))
                assert time.size == 1
                time = time.item()
   
        loaded_label_list_proph.append(loaded_label_proph)

        # --- YOLOX PREDICTIONS ---
        # Assumes batch of post-processed predictions from YoloX Head.
        # See postprocessing: https://github.com/Megvii-BaseDetection/YOLOX/blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/utils/boxes.py#L32
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        num_pred = 0 if yolox_preds is None else yolox_preds.shape[0]
        yolox_pred_proph = np.zeros((num_pred,), dtype=BBOX_DTYPE)
        if num_pred > 0:
            if isinstance(loaded_labels, np.ndarray):
                yolox_preds = yolox_preds
            else:
                yolox_preds = yolox_preds.detach().cpu().numpy()
            assert yolox_preds.shape == (num_pred, 7)
            yolox_pred_proph['t'] = np.ones((num_pred,), dtype=BBOX_DTYPE['t']) * time
            yolox_pred_proph['x'] = np.asarray(yolox_preds[:, 0], dtype=BBOX_DTYPE['x'])
            yolox_pred_proph['y'] = np.asarray(yolox_preds[:, 1], dtype=BBOX_DTYPE['y'])
            yolox_pred_proph['w'] = np.asarray(yolox_preds[:, 2] - yolox_preds[:, 0], dtype=BBOX_DTYPE['w'])
            yolox_pred_proph['h'] = np.asarray(yolox_preds[:, 3] - yolox_preds[:, 1], dtype=BBOX_DTYPE['h'])
            yolox_pred_proph['class_id'] = np.asarray(yolox_preds[:, 6], dtype=BBOX_DTYPE['class_id']) 
            yolox_pred_proph['class_confidence'] = np.asarray(yolox_preds[:, 5], dtype=BBOX_DTYPE['class_confidence'])
        yolox_pred_list_proph.append(yolox_pred_proph)

    return loaded_label_list_proph, yolox_pred_list_proph

