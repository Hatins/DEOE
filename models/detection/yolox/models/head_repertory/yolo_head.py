"""
ours PU-weight
When using that head, please moving the head.py into models/detection/yolox/models and rename it as yolo_head.py
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from models.detection.yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
import ipdb

class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            strides=(8, 16, 32),
            in_channels=(256, 512, 1024),
            act="silu",
            depthwise=False,
            compile_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        #set num_classes = 1 for all the time
        self.decode_in_inference = True  # for deploy, set to False

        self.obj_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.pos_obj_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        self.output_strides = None
        self.output_grids = None

        # Automatic width scaling according to original YoloX channel dims.
        # in[-1]/out = 4/1
        # out = in[-1]/4 = 256 * width
        # -> width = in[-1]/1024
        largest_base_dim_yolox = 1024
        largest_base_dim_from_input = in_channels[-1]
        width = largest_base_dim_from_input/largest_base_dim_yolox

        hidden_dim = int(256*width)

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=in_channels[i],
                    out_channels=hidden_dim,
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.obj_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.pos_obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # According to Focal Loss paper:
        self.initialize_biases(prior_prob=0.01)

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile YOLOXHead because torch.compile is not available')
        ##################################

    def initialize_biases(self, prior_prob):
        for conv in self.pos_obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None):
        train_outputs = []
        inference_outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (obj_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.obj_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            obj_x = x
            reg_x = x

            obj_feat = obj_conv(obj_x)
            pos_obj_output = self.pos_obj_preds[k](obj_feat)
            obj_output = self.obj_preds[k](obj_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)


            if self.training:
                output = torch.cat([reg_output, obj_output, pos_obj_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                train_outputs.append(output)
            inference_output = torch.cat(
                [reg_output, obj_output.sigmoid(), pos_obj_output.sigmoid()], 1
            )
            inference_outputs.append(inference_output)

        # --------------------------------------------------------
        # Modification: return decoded output also during training
        # --------------------------------------------------------
        losses = None
        if self.training:
            losses =  self.get_losses(
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(train_outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
            assert len(losses) == 6
            losses = {
                "loss": losses[0],
                "iou_loss": losses[1],
                "conf_loss": losses[2], # object-ness
                "pos_obj_loss": losses[3], # pos object-ness
                "l1_loss": losses[4],
                "num_fg": losses[5],
            }
        self.hw = [x.shape[-2:] for x in inference_outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in inference_outputs], dim=2
        ).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].type()), losses
        else:
            return outputs, losses
        
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid
    
    def decode_outputs(self, outputs, dtype):
        if self.output_grids is None:
            assert self.output_strides is None
            grids = []
            strides = []
            for (hsize, wsize), stride in zip(self.hw, self.strides):
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                strides.append(torch.full((*shape, 1), stride))

            self.output_grids = torch.cat(grids, dim=1).type(dtype)
            self.output_strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + self.output_grids) * self.output_strides,
            torch.exp(outputs[..., 2:4]) * self.output_strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs
    
    def create_mask(self, input, N_anchors, N_samples):
        N = input.shape[0] // N_anchors
        mask = torch.zeros_like(input, dtype=torch.bool)
        for i in range(N):
            start = i * N_anchors
            end = start + N_anchors
            sub_input = input[start:end].squeeze()
            _, indices = torch.topk(sub_input, N_samples, largest=False)
            sub_mask = torch.zeros_like(sub_input, dtype=torch.bool)
            sub_mask[indices] = True
            mask[start:end] = sub_mask.unsqueeze(1)
        
        return mask
    
    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        pos_obj_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, 1]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        pos_obj_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                pos_obj_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        pos_obj_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise

                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        pos_obj_preds,
                        obj_preds,
                        "cpu",
                    )
                num_fg += num_fg_img

                pos_obj_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )


            reg_targets.append(reg_target)
            pos_obj_targets.append(pos_obj_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        pos_obj_targets = torch.cat(pos_obj_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fg = max(num_fg, 1)

        neg_pred = pos_obj_preds.view(-1,1)[~fg_masks]
        neg_soft = (1 - F.sigmoid(neg_pred)).float()

        obj_weight = torch.zeros_like(pos_obj_preds,dtype=torch.float).view(-1,1)
        # obj_weight[fg_masks] = pos_soft
        obj_weight[~fg_masks] = neg_soft
        obj_weight[fg_masks] = 1

        # PU_mask = (PU_mask.float()).unsqueeze(dim=1)

        loss_obj = ((self.bcewithlog_loss(obj_preds.view(-1, 1), 
                                        obj_targets))*obj_weight).sum() / num_fg
        
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        # loss_obj = (
        #     self.bcewithlog_loss(obj_preds.view(-1, 1)[PU_mask], obj_targets[PU_mask])
        # ).sum() / num_fg
        loss_obj_pos = (
            self.bcewithlog_loss(
                pos_obj_preds.view(-1, 1)[fg_masks], pos_obj_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0

        loss = reg_weight * loss_iou + loss_obj + loss_obj_pos + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_obj_pos,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
    

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
    
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        pos_obj_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )



        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        pos_obj_preds_ = pos_obj_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_pos_obj_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            pos_obj_preds_, obj_preds_ = pos_obj_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            pos_obj_preds_ = (
                pos_obj_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_pos_obj_loss = F.binary_cross_entropy(
                pos_obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_pos_obj_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del pos_obj_preds_

        cost = (
            pair_wise_pos_obj_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_pos_obj_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
    
    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation
    
    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds