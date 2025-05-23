from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
from models.detection.yolox.utils.boxes import postprocess,postprocess2
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee,to_prophesee2 
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, REGStates, Mode, mode_2_string, \
    merge_mixed_batches

import os
import cv2
import ipdb

def remove_elements(ori_items, moving_items):
    return [elem for elem in ori_items if elem not in moving_items]

class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()
        self.full_config = full_config
        self.keep_cls = full_config.model.backbone.keep_cls
        self.memory = full_config.model.backbone.memory_type
        self.mdl_config = full_config.model
        self.head_name = full_config.model.head.name
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }
        
        self.reg_states = REGStates()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.training_classes = self.full_config.dataset.training_classes
            self.unseen_classes = self.full_config.dataset.unseen_classes
            self.testing_classes = self.full_config.dataset.testing_classes
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            #We set two evaluator, one (0) for unseen classes and one (1) for all classes
            self.mode_2_psee_evaluator[Mode.VAL] = [PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2),
                PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
                ]
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            self.unseen_classes = self.full_config.dataset.unseen_classes
            self.testing_classes = self.full_config.dataset.testing_classes
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = [PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2),
                PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
                ]
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.unseen_classes = self.full_config.dataset.unseen_classes
            self.testing_classes = self.full_config.dataset.testing_classes
            self.mode_2_psee_evaluator[Mode.TEST] = [PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2), 
                
                PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            ]
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def set_model_to_gpus(self, device):
        self.mdl = self.mdl.to(device)

    def freeze_model(self):
        for param in self.mdl.parameters():
            param.requires_grad = False

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets=None) \
            -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
                self.started_training = False
                self.training = False
                event_tensor = event_tensor.to(dtype=self.dtype)
                event_tensor = self.input_padder.pad_tensor_ev_repr(event_tensor)

                backbone_features, states = self.mdl.forward_backbone(x=event_tensor, previous_states=previous_states)
                
                predictions, _ = self.mdl.forward_detect(backbone_features=backbone_features)

                pred_processed = postprocess(prediction=predictions,
                                            conf_thre=0.1,
                                            nms_thre=0.05,
                                            mode='train')
                
                # pred_processed = postprocess2(prediction=predictions,
                #                             num_classes=self.mdl_config.head.num_classes,
                #                             conf_thre=0.3,
                #                             nms_thre=self.mdl_config.postprocess.nms_threshold,
                #                             mode='train')

        
                return pred_processed[0], states

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']
    
    def training_step_with_ssm(self, batch: Any, batch_idx: int):
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step

        '''
        batchsize * suqence_length
        '''
        ev_tensor_sequence = data[DataType.EV_REPR] # squence_length * B * H * W
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])

        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()

        if type(self.training_classes) != list:
            self.training_classes = list(self.training_classes.keys())
        else:
            self.training_classes = self.training_classes

        ev_tensor_sequence = torch.stack(
            ev_tensor_sequence
        )  # shape: (sequence_len, batch_size, channels, height, width) = (L, B, C, H, W)

        ev_tensor_sequence = ev_tensor_sequence.to(dtype=self.dtype)
        ev_tensor_sequence = self.input_padder.pad_tensor_ev_repr(ev_tensor_sequence)

        if token_mask_sequence is not None:
            token_mask_sequence = torch.stack(token_mask_sequence)
            token_mask_sequence = token_mask_sequence.to(dtype=self.dtype)
            token_mask_sequence = self.input_padder.pad_token_mask(
                token_mask=token_mask_sequence
            )
        else:
            token_mask_sequence = None

        if self.mode_2_hw[mode] is None:
            self.mode_2_hw[mode] = tuple(ev_tensor_sequence.shape[-2:])
        else:
            assert self.mode_2_hw[mode] == ev_tensor_sequence.shape[-2:]

        backbone_features, states = self.mdl.forward_backbone_ssm(
            ev_input=ev_tensor_sequence,
            previous_states=prev_states,
            token_mask=token_mask_sequence
        )

        prev_states = states

        
        for tidx, curr_labels in enumerate(sparse_obj_labels):
            (
                current_labels,
                valid_batch_indices,
            ) = curr_labels.get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(
                    backbone_features={
                        k: v[tidx] for k, v in backbone_features.items()
                    },
                    selected_indices=valid_batch_indices,
                )
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(
                    event_representations=ev_tensor_sequence[tidx],
                    selected_indices=valid_batch_indices,
                )

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = (
            backbone_feature_selector.get_batched_backbone_features()
        )
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, training_classes = self.training_classes,format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)
        labels_yolox = ObjectLabels.labels_mapping(self.device, labels_yolox)
        predictions, losses = self.mdl.forward_detect(
                   backbone_features=selected_backbone_features, targets=labels_yolox
        )

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM,):      
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(prediction=predictions,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold,
                                     mode='train')

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        assert losses is not None
        assert "loss" in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            "loss": losses["loss"],
        }

        prefix = f"{mode_2_string[mode]}/"
        log_dict = {f"{prefix}{k}": v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if (
                self.train_metrics_config.detection_metrics_every_n_steps is not None
                and step > 0
                and step % self.train_metrics_config.detection_metrics_every_n_steps == 0
            ):
                self.run_psee_evaluator(mode=mode)

        return output
    
    def training_step_with_rnn(self, batch: Any, batch_idx: int):
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step

        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        if type(self.training_classes) != list:
            self.training_classes = list(self.training_classes.keys())
        else:
            self.training_classes = self.training_classes
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states
            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, training_classes = self.training_classes,format_='yolox')

        labels_yolox = labels_yolox.to(dtype=self.dtype)

        labels_yolox = ObjectLabels.labels_mapping(self.device, labels_yolox, keep_cls=self.keep_cls)

        
        predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                      targets=labels_yolox)
        
        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
                # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
                # This is because otherwise we would mostly evaluate the init phase of the sequence.
                predictions = predictions[-batch_size:]
                obj_labels = obj_labels[-batch_size:]

        if not self.keep_cls:
            pred_processed = postprocess(prediction=predictions,
                                        conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                        nms_thre=self.mdl_config.postprocess.nms_threshold,
                                        mode='train')
            
            loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed, keep_classes=self.training_classes)

        else:
            pred_processed = postprocess2(prediction=predictions,
                                        num_classes=self.mdl_config.head.num_classes,
                                        conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                        nms_thre=self.mdl_config.postprocess.nms_threshold,
                                        mode='train')

            loaded_labels_proph, yolox_preds_proph = to_prophesee2(obj_labels, pred_processed, keep_classes=self.training_classes)

        assert losses is not None
        assert 'loss' in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)
        return output
    
    def vis_and_save_image(self, ev_pr, label, pred, unseen_classes, 
                           save_dir = '/home/zht/python_project/RVT_CAOD_v9/save_img/', threshold = 0.3, topn = 10):

        files = os.listdir(save_dir)
        index = len(files)
        ev_pr = ev_pr.to('cpu')
        assert ev_pr.shape[0] % 2 == 0
        num_bins = int(ev_pr.shape[0] / 2)
        height = int(ev_pr.shape[1])
        width = int(ev_pr.shape[2])
        ev_pr = ev_pr.permute(1, 2, 0)
        ev_pr = ev_pr.numpy()
        frame = np.zeros((height, width, 3), dtype=np.uint8) 
        for i in range(num_bins):
            pos_image = (ev_pr[:, :, i + num_bins]).astype(np.uint8)
            neg_image = (ev_pr[:, :, i]).astype(np.uint8)
            pos_image = cv2.equalizeHist(pos_image)
            neg_image = cv2.equalizeHist(neg_image)
            image = np.concatenate((neg_image[..., None], np.zeros((height, width, 1), dtype=np.uint8), pos_image[..., None]), axis=-1)
            frame = np.add(frame, image)  
        frame = frame * 255.0
        frame_copy = frame.copy()
        # topn = label.shape[0]
        fix_num_threshold = np.partition(pred['class_confidence'], -topn)[-topn]
        if fix_num_threshold > threshold:
            pass
        else:
            threshold = fix_num_threshold
        mask = pred['class_confidence'] > threshold
        pred = pred[mask]

        for item in pred:
            x, y, w, h = item['x'], item['y'], item['w'], item['h']
            left = int(x)
            top = int(y) 
            right = int(x + w)
            bottom = int(y + h)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 250, 250), 1)
        
        for item in label:
            x, y, w, h = item['x'], item['y'], item['w'], item['h']
            class_id = item['class_id']

            left = int(x)
            top = int(y) 
            right = int(x + w)
            bottom = int(y + h)
            center = ((left + right) // 2, (top + bottom) // 2)
            if class_id in unseen_classes:
                color = (255, 165, 0)  
                cv2.putText(frame_copy, str(class_id), (center[0], bottom - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 1)

        stacked_image = cv2.hconcat([frame, frame_copy])
        save_path = save_dir + '{}.png'.format(index)
        cv2.imwrite(save_path, stacked_image)



    def concatenate_tensors(self, tensor1, tensor2, order1, order2):

        D1 = tensor1.shape[0]
        D2 = tensor2.shape[0]
        D = D1 + D2

        result_shape = (D,) + tensor1.shape[1:]
        result = torch.zeros(result_shape, dtype=tensor1.dtype).to(tensor1.device)

        for i, idx in enumerate(order1):
            result[idx] = tensor1[i]
    
        for i, idx in enumerate(order2):
            result[idx] = tensor2[i]

        return result
    
    def subtract_lists(self, listA: list, listB: list) -> list:
        return [x for x in listA if x not in listB]
    

    def merge_dicts_and_average(self, dicts_list: list):
        result_dict = {}
        num_dicts = len(dicts_list)
        
        for d in dicts_list:
            for key, value in d.items():
                if key in result_dict:
                    result_dict[key] += value
                else:
                    result_dict[key] = value

        for key in result_dict:
            result_dict[key] /= num_dicts
        
        return result_dict
    
    def training_step_without_pre_reg(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if self.memory == 'lstm':
            return self.training_step_with_rnn(batch, batch_idx=batch_idx)
        elif self.memory == 's5':
            return self.training_step_with_ssm(batch, batch_idx=batch_idx)
    
    def training_step_with_pre_reg(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        self.reg_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        
        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

        prev_reg = self.reg_states.get_states(worker_id=worker_id) #loading the labels in last time
        
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        predictions_list = list()
        losses_list = list()

        if type(self.training_classes) != list:
            self.training_classes = list(self.training_classes.keys())
        else:
            self.training_classes = self.training_classes
            
        first_valid_flag = True

        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states
    
            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            inference_valid = self.subtract_lists(list(range(batch_size)), valid_batch_indices) #Find the samples in a batch without valid labels.
 
            #process the samples with the corresponding labels
            if len(current_labels) > 0:  #We should predict the results step one step to provide the 'prev_reg'
                backbone_feature_selector = BackboneFeatureSelector()
  
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                            selected_indices=valid_batch_indices)


                selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
                #get the label
                labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=current_labels, training_classes = self.training_classes,format_='yolox')
                labels_yolox = labels_yolox.to(dtype=self.dtype)
                labels_yolox = ObjectLabels.labels_mapping(self.device, labels_yolox) #set the bbox, which do not belong to the training classes, to 0

                #get the single output
                if len(prev_reg) > 0:
                    prev_reg = prev_reg[valid_batch_indices] #find the corresponding prev_reg.

                
                prev_first_sample_flag = is_first_sample[valid_batch_indices] #find whether this sequence is the first sequence
                if first_valid_flag == False:
                    prev_first_sample_flag = [False for _ in prev_first_sample_flag] #if it's not the start of a sequence, set all the prev_first_sample_flag to false
                prev_reg_list = [prev_first_sample_flag, prev_reg] #[whether is the first samples, the prev_reg]
               
                predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                        targets=labels_yolox, prev_reg=prev_reg_list)
           
                predictions_list.append(predictions)
                losses_list.append(losses)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)
            #process the samples without the corresponding labels
            if len(inference_valid)>0: #the results of those samples without the corresponding labels, we still need to get the results. 
                
                backbone_feature_selector = BackboneFeatureSelector()

                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                            selected_indices=inference_valid)

                selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()

                self.eval()
                predictions_rest, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)
                self.train()

            #concat the two results to provide the prev_reg for the next time
            if len(current_labels) > 0 and len(inference_valid) > 0:

                prev_reg = self.concatenate_tensors(predictions, predictions_rest, valid_batch_indices, inference_valid)
            elif len(current_labels) > 0 and len(inference_valid) == 0:
                prev_reg = predictions #all samples have the corrsponding labels
            elif len(current_labels) == 0 and len(inference_valid) > 0:
                prev_reg = predictions_rest #no samples have the corrsponding labels

            first_valid_flag = False

            prev_reg = prev_reg[:,:,0:4] #we do not need the confidence for temporal IoU, so we discard it

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
   
        self.reg_states.save_states_and_detach(worker_id=worker_id, prev_reg = prev_reg)

        predictions = torch.cat(predictions_list,dim=0) #the results for the samples has the corrsponding labels in current time.

        losses = self.merge_dicts_and_average(losses_list) #get the average loss
 
        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]


        pred_processed = postprocess(prediction=predictions,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold,
                                     mode='train')
         
        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed, keep_classes=self.training_classes)
        assert losses is not None
        assert 'loss' in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # output = {
        #     ObjDetOutput.SKIP_VIZ: True,
        #     'loss': torch.tensor(0.0, requires_grad=True).to(predictions.device)
        # }


        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        
        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)

        return output

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if self.head_name in ['DEOE', 'dual_regressor_head']:
            return self.training_step_with_pre_reg(batch=batch, batch_idx=batch_idx)
        else:
            return self.training_step_without_pre_reg(batch=batch, batch_idx=batch_idx)

    def _val_test_step_impl_with_ssm(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)

        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])

        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()

        if type(self.unseen_classes) != list:
            self.unseen_classes = list(self.unseen_classes.keys())
        else:
            self.unseen_classes = self.unseen_classes

        if type(self.testing_classes) != list:
            self.testing_classes = list(self.testing_classes.keys())
        else:
            self.testing_classes = self.testing_classes

        ev_tensor_sequence = torch.stack(
            ev_tensor_sequence
        )  # shape: (sequence_len, batch_size, channels, height, width) = (L, B, C, H, W)

        ev_tensor_sequence = ev_tensor_sequence.to(dtype=self.dtype)

        ev_tensor_sequence = self.input_padder.pad_tensor_ev_repr(ev_tensor_sequence)

        if self.mode_2_hw[mode] is None:
            self.mode_2_hw[mode] = tuple(ev_tensor_sequence.shape[-2:])
        else:
            assert self.mode_2_hw[mode] == ev_tensor_sequence.shape[-2:]

        backbone_features, states = self.mdl.forward_backbone_ssm(ev_input=ev_tensor_sequence, previous_states=prev_states)
        prev_states = states

        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or (
                self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM
            )

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(
                        backbone_features={
                            k: v[tidx] for k, v in backbone_features.items()
                        },
                        selected_indices=valid_batch_indices,
                    )

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(
                        event_representations=ev_tensor_sequence[tidx],
                        selected_indices=valid_batch_indices,
                    )
   
        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states
        )

        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        #When the mode is set to "val," we implemented certain strategies to ensure that the final 
        #output quantity exceeds 800 for the purpose of measuring AR@300.
        pred_processed = postprocess(prediction=predictions,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold,
                                     mode='val')
        

        loaded_labels_proph_unseen, yolox_preds_proph_unseen = to_prophesee(obj_labels, 
                                                            pred_processed, keep_classes=self.unseen_classes)
        # For visualization, we only use the last item (per batch).
        empty_index = []
        #find out which index doesn't contain any of target categories for visualization.
        for index in range(len(loaded_labels_proph_unseen)):
            save_flag = True
            for i in range(len(loaded_labels_proph_unseen[index])):
                if loaded_labels_proph_unseen[index][i][6] != 1:
                    save_flag = False
            if save_flag == True:
                empty_index.append(index)

        existing_items = remove_elements(list(range(0,len(loaded_labels_proph_unseen))),empty_index)
        if len(empty_index) < len(loaded_labels_proph_unseen):

            loaded_labels_proph_unseen_selected = [loaded_labels_proph_unseen[i] for 
                                i in range(len(loaded_labels_proph_unseen)) if i not in empty_index]
            
            yolox_preds_proph_unseen_selected = [yolox_preds_proph_unseen[i] for 
                                i in range(len(yolox_preds_proph_unseen)) if i not in empty_index]

            output = {
                        ObjDetOutput.LABELS_PROPH: loaded_labels_proph_unseen_selected[-1],
                        ObjDetOutput.PRED_PROPH: yolox_preds_proph_unseen_selected[-1],
                        ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(
                start_idx=existing_items[-1])[0],
                        ObjDetOutput.SKIP_VIZ: False,
                    }
        else:
            output = {
                        ObjDetOutput.SKIP_VIZ: True,
                    }
            
        if mode == Mode.TEST and output[ObjDetOutput.SKIP_VIZ] == False:
            self.vis_and_save_image(output[ObjDetOutput.EV_REPR], output[ObjDetOutput.LABELS_PROPH],
                                    output[ObjDetOutput.PRED_PROPH], self.unseen_classes)

        if self.started_training:
            self.mode_2_psee_evaluator[mode][0].add_labels(loaded_labels_proph_unseen)
            self.mode_2_psee_evaluator[mode][0].add_predictions(yolox_preds_proph_unseen)
            #In order to ensure a relative balance between seen and unseen samples
            #We only load the images which contain unseen samples for evaluation
            self.mode_2_psee_evaluator[mode][1].add_labels(loaded_labels_proph_unseen)
            self.mode_2_psee_evaluator[mode][1].add_predictions(yolox_preds_proph_unseen)
            self.mode_2_psee_evaluator[mode][1].set_ignored_to_False()
        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()

        if type(self.unseen_classes) != list:
            self.unseen_classes = list(self.unseen_classes.keys())
        else:
            self.unseen_classes = self.unseen_classes

        if type(self.testing_classes) != list:
            self.testing_classes = list(self.testing_classes.keys())
        else:
            self.testing_classes = self.testing_classes
            
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)

                    obj_labels.extend(current_labels)

                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        #When the mode is set to "val," we implemented certain strategies to ensure that the final 
        #output quantity exceeds 800 for the purpose of measuring AR@300.
        if not self.keep_cls:
            pred_processed = postprocess(prediction=predictions,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold,
                                     mode='val')

        
            loaded_labels_proph_unseen, yolox_preds_proph_unseen = to_prophesee(obj_labels, 
                                                            pred_processed, keep_classes=self.unseen_classes)
        
        else:
            pred_processed = postprocess2(prediction=predictions,
                                    num_classes=self.mdl_config.head.num_classes,
                                    conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                    nms_thre=self.mdl_config.postprocess.nms_threshold,
                                    mode='val')

            loaded_labels_proph_unseen, yolox_preds_proph_unseen = to_prophesee2(obj_labels, 
                                                            pred_processed, keep_classes=self.unseen_classes)

        # For visualization, we only use the last item (per batch).
        empty_index = []
        #find out which index doesn't contain any of target categories for visualization.
        for index in range(len(loaded_labels_proph_unseen)):
            save_flag = True
            for i in range(len(loaded_labels_proph_unseen[index])):
                if loaded_labels_proph_unseen[index][i][6] != 1:
                    save_flag = False
            if save_flag == True:
                empty_index.append(index)

        existing_items = remove_elements(list(range(0,len(loaded_labels_proph_unseen))),empty_index)
        if len(empty_index) < len(loaded_labels_proph_unseen):

            loaded_labels_proph_unseen_selected = [loaded_labels_proph_unseen[i] for 
                                i in range(len(loaded_labels_proph_unseen)) if i not in empty_index]
            
            yolox_preds_proph_unseen_selected = [yolox_preds_proph_unseen[i] for 
                                i in range(len(yolox_preds_proph_unseen)) if i not in empty_index]

            output = {
                        ObjDetOutput.LABELS_PROPH: loaded_labels_proph_unseen_selected[-1],
                        ObjDetOutput.PRED_PROPH: yolox_preds_proph_unseen_selected[-1],
                        ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(
                start_idx=existing_items[-1])[0],
                        ObjDetOutput.SKIP_VIZ: False,
                    }
        else:
            output = {
                        ObjDetOutput.SKIP_VIZ: True,
                    }
            
        if mode == Mode.TEST and output[ObjDetOutput.SKIP_VIZ] == False:
            self.vis_and_save_image(output[ObjDetOutput.EV_REPR], output[ObjDetOutput.LABELS_PROPH],
                                    output[ObjDetOutput.PRED_PROPH], self.unseen_classes)

        if self.started_training:
            self.mode_2_psee_evaluator[mode][0].add_labels(loaded_labels_proph_unseen)
            self.mode_2_psee_evaluator[mode][0].add_predictions(yolox_preds_proph_unseen)
            #In order to ensure a relative balance between seen and unseen samples
            #We only load the images which contain unseen samples for evaluation
            self.mode_2_psee_evaluator[mode][1].add_labels(loaded_labels_proph_unseen)
            self.mode_2_psee_evaluator[mode][1].add_predictions(yolox_preds_proph_unseen)
            self.mode_2_psee_evaluator[mode][1].set_ignored_to_False()
        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        if self.memory in ['lstm', 'gru', 'rnn']:
            return self._val_test_step_impl(batch, mode=Mode.VAL)
        elif self.memory in ['s4', 's5', 's6']:
            return self._val_test_step_impl_with_ssm(batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        if self.memory in ['lstm', 'gru', 'rnn']:
            return self._val_test_step_impl(batch, mode=Mode.TEST)
        elif self.memory in ['s4', 's5', 's6']:
            return self._val_test_step_impl_with_ssm(batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):

        for eva_index, psee_evaluator in enumerate(self.mode_2_psee_evaluator[mode]):
            if eva_index == 0:
                suffix = '_unseen'
            elif eva_index == 1:
                suffix = ''
            batch_size = self.mode_2_batch_size[mode]
            hw_tuple = self.mode_2_hw[mode]
            if psee_evaluator is None:
                warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
                return
            assert batch_size is not None
            assert hw_tuple is not None
            if psee_evaluator.has_data():
                metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                        img_width=hw_tuple[1])
                assert metrics is not None
                prefix = f'{mode_2_string[mode]}/'
                step = self.trainer.global_step
                log_dict = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        value = torch.tensor(v)
                    elif isinstance(v, np.ndarray):
                        value = torch.from_numpy(v)
                    elif isinstance(v, torch.Tensor):
                        value = v
                    else:
                        raise NotImplementedError
                    assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                    # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                    log_dict[f'{prefix}{k}{suffix}'] = value.to(self.device)
                # Somehow self.log does not work when we eval during the training epoch.
                self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                if dist.is_available() and dist.is_initialized():
                    # We now have to manually sync (average the metrics) across processes in case of distributed training.
                    # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                    # and wandb metric:
                    # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                    # - wandb uses log_metrics that we reduce manually to global rank 0
                    dist.barrier()
                    for k, v in log_dict.items():
                        dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                        if dist.get_rank() == 0:
                            log_dict[k] /= dist.get_world_size()
                if self.trainer.is_global_zero:
                    # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                    # I might not understand wandb login correctly. This works reasonably well for now.
                    add_hack = 2
                    self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

                psee_evaluator.reset_buffer()
            else:
                warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_metrics_config.detection_metrics_every_n_steps is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode][0].has_data()
            assert self.mode_2_psee_evaluator[mode][1].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode][0].has_data()
        assert self.mode_2_psee_evaluator[mode][1].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
