# -*- coding: utf-8 -*-
# @Description: Main network architecture for GeoMVSNet.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from models.submodules import homo_warping, init_inverse_range, schedule_inverse_range, FPN, Reg2d
from models.geometry import GeoFeatureFusion, GeoRegNet2d
from models.filter import frequency_domain_filter
from .FMT import FMT_with_pathway

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


class GeoMVSNet(nn.Module):
    def __init__(self, levels, hypo_plane_num_stages, depth_interal_ratio_stages,
                 feat_base_channel, reg_base_channel, group_cor_dim_stages, var_reg_dim_stages):
        super(GeoMVSNet, self).__init__()

        self.levels = levels
        self.hypo_plane_num_stages = hypo_plane_num_stages
        self.depth_interal_ratio_stages = depth_interal_ratio_stages

        self.StageNet = StageNet()
        self.FMT_with_pathway = FMT_with_pathway()

        # feature settings
        self.FeatureNet = FPN(base_channels=feat_base_channel)
        self.coarest_separate_flag = True
        if self.coarest_separate_flag:
            self.CoarestFeatureNet = FPN(base_channels=feat_base_channel)
        self.GeoFeatureFusionNet = GeoFeatureFusion(
            convolutional_layer_encoding="z", mask_type="basic", add_origin_feat_flag=True)

        # cost regularization settings
        self.RegNet_stages = nn.ModuleList()
        self.var_RegNet_stages = nn.ModuleList()
        self.group_cor_dim_stages = group_cor_dim_stages
        self.geo_reg_flag = True
        self.geo_reg_encodings = ['std', 'z', 'z', 'z']  # must use std in idx-0
        for stage_idx in range(self.levels):
            in_dim = group_cor_dim_stages[stage_idx]
            var_reg_dim = var_reg_dim_stages[stage_idx]
            if self.geo_reg_flag:
                self.RegNet_stages.append(GeoRegNet2d(input_channel=in_dim, base_channel=reg_base_channel,
                                                      convolutional_layer_encoding=self.geo_reg_encodings[stage_idx]))
                self.var_RegNet_stages.append(GeoRegNet2d(input_channel=var_reg_dim, base_channel=reg_base_channel,
                                                          convolutional_layer_encoding=self.geo_reg_encodings[stage_idx]))
            else:
                self.RegNet_stages.append(Reg2d(input_channel=in_dim, base_channel=reg_base_channel))

        # frequency domain filter settings
        self.curriculum_learning_rho_ratios = [9, 4, 2, 1]

    def get_mask(self, stage_name, ref_lbp_mask, confidence_last, mask):
        lbp_mask_0 = ref_lbp_mask[stage_name][0]
        lbp_mask_1 = ref_lbp_mask[stage_name][1]
        lbp_mask_2 = ref_lbp_mask[stage_name][2]
        lbp_mask_3 = ref_lbp_mask[stage_name][3]
        lbp_mask_4 = ref_lbp_mask[stage_name][4]
        lbp_mask_5 = ref_lbp_mask[stage_name][5]
        lbp_mask_6 = ref_lbp_mask[stage_name][6]
        lbp_mask_7 = ref_lbp_mask[stage_name][7]
        lbp_mask_8 = ref_lbp_mask[stage_name][8]
        mask_texture = ((lbp_mask_0 + lbp_mask_1 + lbp_mask_2 + lbp_mask_3) != 0)
        ratio = torch.sum(mask_texture) / (mask_texture.shape[1] * mask_texture.shape[2])

        if stage_name == "stage1":
            final_mask = mask_texture.float()
            final_mask = torch.logical_and(final_mask, mask)
            ratio_final = torch.sum(final_mask) / (final_mask.shape[1] * final_mask.shape[2])

        elif stage_name == "stage2":
            confidence_last = confidence_last.squeeze(1)
            conf_mean = torch.mul(confidence_last, mask)

            conf_mean = torch.sum(conf_mean, dim=[1, 2])
            valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-12
            conf_mean = torch.mean(conf_mean / valid_pixel_num)
            confidence_mask_2 = confidence_last < conf_mean

            ratio_1 = torch.sum(confidence_mask_2) / (confidence_mask_2.shape[1] * confidence_mask_2.shape[2])

            final_mask = torch.logical_and(confidence_mask_2, mask_texture)

            final_mask = torch.logical_and(final_mask.float(),mask)

            ratio_final = torch.sum(final_mask) / (final_mask.shape[1] * final_mask.shape[2])
        elif stage_name == "stage3":
            confidence_last = confidence_last.squeeze(1)
            conf_mean = torch.mul(confidence_last, mask)
            conf_mean = torch.sum(conf_mean, dim=[1, 2])
            valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-12
            conf_mean = torch.mean(conf_mean / valid_pixel_num)
            confidence_mask_3 = (confidence_last < conf_mean)

            ratio_1 = torch.sum(confidence_mask_3) / (confidence_mask_3.shape[1] * confidence_mask_3.shape[2])

            final_mask = torch.logical_and(confidence_mask_3, mask_texture)

            final_mask = torch.logical_and(final_mask.float(),mask)

            ratio_final = torch.sum(final_mask) / (final_mask.shape[1] * final_mask.shape[2])

        elif stage_name == "stage4":
            confidence_last = confidence_last.squeeze(1)
            conf_mean = torch.mul(confidence_last, mask)
            conf_mean = torch.sum(conf_mean, dim=[1, 2])
            valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-12
            conf_mean = torch.mean(conf_mean / valid_pixel_num)
            confidence_mask_4 = confidence_last < 0.3* conf_mean

            ratio_1 = torch.sum(confidence_mask_4) / (confidence_mask_4.shape[1] * confidence_mask_4.shape[2])

            final_mask = torch.logical_and(confidence_mask_4, mask_texture)

            final_mask = torch.logical_and(final_mask.float(), mask)

            ratio_final = torch.sum(final_mask) / (final_mask.shape[1] * final_mask.shape[2])

        return final_mask

    def fill_in_fast(self, depth_filter_, max_depth=1.00001, custom_kernel=DIAMOND_KERNEL_5,
                     blur_type='gaussian'):
        """Fast, in-place depth completion.

        Args:
            depth_map: projected depths
            max_depth: max depth value for inversion
            custom_kernel: kernel to apply initial dilation
            extrapolate: whether to extrapolate by extending depths to top of
                the frame, and applying a 31x31 full kernel dilation
            blur_type:
                'bilateral' - preserves local structure (recommended)
                'gaussian' - provides lower RMSE

        Returns:
            depth_map: dense depth map
        """
        depth_filter = depth_filter_.clone()
        depth_filter = depth_filter.squeeze(1)
        depth_filter_np = depth_filter.cpu().detach().numpy()
        depth_map = depth_filter_np.copy()
        depth_map = depth_map.astype(np.float32)

        mask_depth1 = (depth_map > 1.0)
        ratio_11 = np.sum(mask_depth1) / (mask_depth1.shape[1] * mask_depth1.shape[2])
        depth_map[mask_depth1] = 0.0

        mask_1 = (depth_map > 1.0)
        num_1 = np.sum(mask_1)

        mask_depth2 = (depth_map < 0.0)
        ratio_12 = np.sum(mask_depth2) / (mask_depth2.shape[1] * mask_depth2.shape[2])
        depth_map[mask_depth2] = 0.0

        mask_2 = (depth_map < 0.0)
        num_2 = np.sum(mask_2)

        # Invert
        valid_pixels = (depth_map > 0.0)
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

        mask_1 = (depth_map>max_depth)
        num_1 = np.sum(mask_1)

        mask_2 = (depth_map<0.0)
        num_2 = np.sum(mask_2)



        # Dilate
        depth_map = cv2.dilate(depth_map, custom_kernel, iterations=2)

        # Hole closing
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

        # Fill empty spaces with dilated values
        empty_pixels = (depth_map == 0.0)
        dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
        depth_map[empty_pixels] = dilated[empty_pixels]

        empty_pixels = (depth_map == 0.0)
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

        # Median blur
        depth_map = cv2.medianBlur(depth_map, 5)

        # Bilateral or Gaussian blur
        if blur_type == 'bilateral':
            # Bilateral blur
            depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        elif blur_type == 'gaussian':
            # Gaussian blur
            valid_pixels = (depth_map > 0.0)
            blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
            depth_map[valid_pixels] = blurred[valid_pixels]

        # Invert
        valid_pixels = (depth_map > 0.0)
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
        depth_filter_ = torch.from_numpy(depth_map).cuda().unsqueeze(1)
        return depth_filter_

    def forward(self, imgs, proj_matrices, intrinsics_matrices, depth_values, ref_lbp_mask, mask, filename=None):

        features = []
        if self.coarest_separate_flag:
            coarsest_features = []
        for nview_idx in range(len(imgs)):
            img = imgs[nview_idx]
            features.append(self.FeatureNet(img))  # B C H W
            if self.coarest_separate_flag:
                coarsest_features.append(self.CoarestFeatureNet(img))


        features = self.FMT_with_pathway(features)

        # coarse-to-fine
        outputs = {}
        for stage_idx in range(self.levels):
            stage_name = "stage{}".format(stage_idx + 1)
            B, C, H, W = features[0][stage_name].shape
            proj_matrices_stage = proj_matrices[stage_name]
            intrinsics_matrices_stage = intrinsics_matrices[stage_name]
            mask_ = mask[stage_name]
            # @Note features
            if stage_idx == 0:
                confidence_last = None
                text_mask = self.get_mask(stage_name, ref_lbp_mask, confidence_last, mask_)
                if self.coarest_separate_flag:
                    features_stage = [feat[stage_name] for feat in coarsest_features]
                else:
                    features_stage = [feat[stage_name] for feat in features]
            elif stage_idx >= 1:
                features_stage = [feat[stage_name] for feat in features]

                ref_img_stage = F.interpolate(imgs[0], size=None, scale_factor=1. / 2 ** (3 - stage_idx),
                                              mode="bilinear", align_corners=False)
                depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear",
                                           align_corners=False)
                confidence_last = F.interpolate(confidence_last.unsqueeze(1), size=None, scale_factor=2,
                                                mode="bilinear", align_corners=False)

                text_mask = self.get_mask(stage_name, ref_lbp_mask, confidence_last, mask_)

                # reference feature
                features_stage[0], refine_depth_map = self.GeoFeatureFusionNet(
                    ref_img_stage, depth_last, confidence_last, depth_values,
                    stage_idx, features_stage[0], intrinsics_matrices_stage,
                    text_mask
                )

            # @Note depth hypos
            if stage_idx == 0:
                depth_hypo = init_inverse_range(depth_values, self.hypo_plane_num_stages[stage_idx], img[0].device,
                                                img[0].dtype, H, W)
            else:
                inverse_min_depth, inverse_max_depth = outputs_stage['inverse_min_depth'].detach(), outputs_stage[
                    'inverse_max_depth'].detach()
                depth_hypo = schedule_inverse_range(inverse_min_depth, inverse_max_depth,
                                                    self.hypo_plane_num_stages[stage_idx], H, W)  # B D H W

            # @Note cost regularization
            geo_reg_data = {}
            if self.geo_reg_flag:
                geo_reg_data['depth_values'] = depth_values
                if stage_idx >= 1 and self.geo_reg_encodings[stage_idx] == 'z':
                    prob_volume_last = F.interpolate(prob_volume_last, size=None, scale_factor=2, mode="bilinear",
                                                     align_corners=False)
                    geo_reg_data["prob_volume_last"] = prob_volume_last

            outputs_stage = self.StageNet(
                stage_idx, features_stage, proj_matrices_stage, depth_hypo=depth_hypo,
                regnet=self.RegNet_stages[stage_idx], var_reg=self.var_RegNet_stages[stage_idx],
                group_cor_dim=self.group_cor_dim_stages[stage_idx],
                depth_interal_ratio=self.depth_interal_ratio_stages[stage_idx],
                geo_reg_data=geo_reg_data, text_mask=text_mask
            )

            # 为下一阶段更新参数，并保持这一阶段的结果
            depth_last = outputs_stage['depth']
            confidence_last = outputs_stage['photometric_confidence']

            depth_est_filtered = frequency_domain_filter(depth_last,
                                                         rho_ratio=self.curriculum_learning_rho_ratios[stage_idx])
            outputs_stage['depth_filtered'] = depth_est_filtered

            # 过滤深度图中mask的地方，使用传统方法进行填充
            depth_min, depth_max = depth_values[:, 0, None, None, None], depth_values[:, -1, None, None, None]
            d = (depth_est_filtered - depth_min) / (depth_max - depth_min)
            text_mask = self.get_mask(stage_name, ref_lbp_mask, confidence_last, mask_).unsqueeze(1)
            depth_filter = torch.mul(d, ~text_mask)
            # 统计弱纹理的概率
            ratio_text_True = torch.sum(text_mask) / (text_mask.shape[2] * text_mask.shape[3])
            if ratio_text_True != 0 and (stage_idx == 0 or stage_idx == 1):
                filter_depth_map = self.fill_in_fast(depth_filter)
                ratio_not_fill = torch.sum(filter_depth_map[text_mask] == 0)/torch.sum(text_mask)
                filter_depth_map = filter_depth_map * (depth_max - depth_min) + depth_min
                depth_est_filtered = filter_depth_map.squeeze(1)
            if stage_idx != 0:
                outputs_stage["depth_refine"] = refine_depth_map.squeeze(1)
            depth_last = depth_est_filtered

            prob_volume_last = outputs_stage['prob_volume']

            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


class StageNet(nn.Module):
    def __init__(self, attn_temp=2):
        super(StageNet, self).__init__()
        self.attn_temp = attn_temp

    def forward(self, stage_idx, features, proj_matrices, depth_hypo, regnet,
                group_cor_dim, depth_interal_ratio, geo_reg_data=None):

        # @Note step1: feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        # @Note step2: cost aggregation
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        cor_weight_sum = 1e-8
        cor_feats = 0
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            save_fn = None
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            del warped_src, src_proj, src_fea

            cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
            cor_weight_sum += cor_weight  # B D H W
            cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat

        cost_volume = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W
        del cor_weight_sum, src_features

        # @Note step3: cost regularization
        if geo_reg_data == {}:
            # basic
            cost_reg = regnet(cost_volume)
        else:
            # probability volume geometry embedding
            cost_reg = regnet(cost_volume, stage_idx, geo_reg_data)
        del cost_volume
        prob_volume = F.softmax(cost_reg, dim=1)  # B D H W

        # @Note step4: depth regression
        prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W

        with torch.no_grad():
            photometric_confidence = prob_volume.max(1)[0]  # B H W
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear',
                                                   align_corners=True).squeeze(1)

        last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
        inverse_min_depth = 1 / depth + depth_interal_ratio * last_depth_itv  # B H W
        inverse_max_depth = 1 / depth - depth_interal_ratio * last_depth_itv  # B H W

        output_stage = {
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            "depth_hypo": depth_hypo,
            "prob_volume": prob_volume,
            "inverse_min_depth": inverse_min_depth,
            "inverse_max_depth": inverse_max_depth,
        }
        return output_stage
