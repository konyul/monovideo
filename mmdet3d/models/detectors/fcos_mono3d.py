# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector
from ..builder import build_voxel_encoder
import numpy as np
import torch
from mmdet3d.core import bbox3d2result

@DETECTORS.register_module()
class FCOSMono3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

#chgd
@DETECTORS.register_module()
class FCOSMonoTemporalI3D(FCOSMono3D):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSMonoTemporalI3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        B = img.size(0)
        img = torch.transpose(img, 1, 2).contiguous()
        img_feats = self.backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_neck:
            img_feats = self.neck(img_feats)
        return img_feats


#chgd
@DETECTORS.register_module()
class FCOSMonoTemporal3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 voxel_encoder,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSMonoTemporal3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, w, l, h, theta, vx, vy] format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore)
        return losses

    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        B = img.size(0)
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        # import cv2
        # import os
        # idx = str(len(os.listdir("augmented_images"))).zfill(6)
        # image_1 = img[0][0].permute(1,2,0).cpu().numpy()
        # image_2 = img[0][1].permute(1,2,0).cpu().numpy()
        # image_3 = img[0][2].permute(1,2,0).cpu().numpy()
        # image_4 = np.hstack((image_3,image_2,image_1))
        # cv2.imwrite(f"augmented_images/{idx}.jpg",image_4)
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        if self.with_neck:
            img_feats = self.neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
        

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        feats = self.extract_feats(imgs, img_metas)
        feats = self.voxel_encoder(feats)
        # only support aug_test for one sample
        outs_list = [self.bbox_head(x) for x in feats]
        for i, img_meta in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):  # for each prediction
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        # every stride of featmap
                        outs_list[i][j][k] = torch.flip(
                            outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    # offset_x
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    # velo_x
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    # rotation
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi

        merged_outs = []
        for i in range(len(outs_list[0])):  # for each prediction
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                # for each stride of featmap
                avg_feats = torch.mean(
                    torch.cat([x[i][j] for x in outs_list]),
                    dim=0,
                    keepdim=True)
                if i == 1:  # regression predictions
                    # rot/velo/2d det keeps the original
                    avg_feats[:, 6:, :, :] = \
                        outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    # dir_cls keeps the original
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)

        bbox_outputs = self.bbox_head.get_bboxes(
            *merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])

        return [bbox_list]



@DETECTORS.register_module()
class FCOSFreezeMonoTemporal3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501


    def __init__(self,
                 backbone,
                 neck,
                 voxel_encoder,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSFreezeMonoTemporal3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, w, l, h, theta, vx, vy] format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        with torch.no_grad():
            x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore)
        return losses

    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        B = img.size(0)
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        # import cv2
        # import os
        # idx = str(len(os.listdir("augmented_images"))).zfill(6)
        # image_1 = img[0][0].permute(1,2,0).cpu().numpy()
        # image_2 = img[0][1].permute(1,2,0).cpu().numpy()
        # image_3 = img[0][2].permute(1,2,0).cpu().numpy()
        # image_4 = np.hstack((image_3,image_2,image_1))
        # cv2.imwrite(f"augmented_images/{idx}.jpg",image_4)
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
    
        if self.with_neck:
            img_feats = self.neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        feats = self.extract_feats(imgs, img_metas)
        feats = self.voxel_encoder(feats)
        # only support aug_test for one sample
        outs_list = [self.bbox_head(x) for x in feats]
        for i, img_meta in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):  # for each prediction
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        # every stride of featmap
                        outs_list[i][j][k] = torch.flip(
                            outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    # offset_x
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    # velo_x
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    # rotation
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi

        merged_outs = []
        for i in range(len(outs_list[0])):  # for each prediction
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                # for each stride of featmap
                avg_feats = torch.mean(
                    torch.cat([x[i][j] for x in outs_list]),
                    dim=0,
                    keepdim=True)
                if i == 1:  # regression predictions
                    # rot/velo/2d det keeps the original
                    avg_feats[:, 6:, :, :] = \
                        outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    # dir_cls keeps the original
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)

        bbox_outputs = self.bbox_head.get_bboxes(
            *merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])

        return [bbox_list]  

@DETECTORS.register_module()
class FCOSPrevFreezeMonoTemporal3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501


    def __init__(self,
                 backbone,
                 neck,
                 voxel_encoder,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSPrevFreezeMonoTemporal3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, w, l, h, theta, vx, vy] format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore)
        return losses

    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        B = img.size(0)
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        # import cv2
        # import os
        # idx = str(len(os.listdir("augmented_images"))).zfill(6)
        # image_1 = img[0][0].permute(1,2,0).cpu().numpy()
        # image_2 = img[0][1].permute(1,2,0).cpu().numpy()
        # image_3 = img[0][2].permute(1,2,0).cpu().numpy()
        # image_4 = np.hstack((image_3,image_2,image_1))
        # cv2.imwrite(f"augmented_images/{idx}.jpg",image_4)
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
            
        
        
        for batch in range(B):
            if batch == 0:
                cur_image = img[batch*N].unsqueeze(0)
                prev_image = img[batch*N+1:batch*N+3]
            else:
                cur_image = torch.cat([cur_image,img[batch*N].unsqueeze(0)],dim=0)
                prev_image = torch.cat([prev_image,img[batch*N+1:batch*N+3]],dim=0)
                
        cur_image_feature = self.backbone(cur_image)
        with torch.no_grad():
            prev_image_feature = self.backbone(prev_image)

        if self.with_neck:
            cur_image_feature = self.neck(cur_image_feature)
            with torch.no_grad():
                prev_image_feature = self.neck(prev_image_feature)
            
        
        
        cur_img_feats_reshaped = []
        prev_img_features_reshaped= []
        for cur_img_feat in cur_image_feature:
            BN, C, H, W = cur_img_feat.size()
            cur_img_feats_reshaped.append(cur_img_feat.view(B, int(BN / B), C, H, W))
        
        for img_feat in prev_image_feature:
            BN, C, H, W = img_feat.size()
            prev_img_features_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        
        return [cur_img_feats_reshaped, prev_img_features_reshaped]
    

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img, img_metas)
        x = self.voxel_encoder(x)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        feats = self.extract_feats(imgs, img_metas)
        feats = self.voxel_encoder(feats)
        # only support aug_test for one sample
        outs_list = [self.bbox_head(x) for x in feats]
        for i, img_meta in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):  # for each prediction
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        # every stride of featmap
                        outs_list[i][j][k] = torch.flip(
                            outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    # offset_x
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    # velo_x
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    # rotation
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi

        merged_outs = []
        for i in range(len(outs_list[0])):  # for each prediction
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                # for each stride of featmap
                avg_feats = torch.mean(
                    torch.cat([x[i][j] for x in outs_list]),
                    dim=0,
                    keepdim=True)
                if i == 1:  # regression predictions
                    # rot/velo/2d det keeps the original
                    avg_feats[:, 6:, :, :] = \
                        outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    # dir_cls keeps the original
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)

        bbox_outputs = self.bbox_head.get_bboxes(
            *merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])

        return [bbox_list]