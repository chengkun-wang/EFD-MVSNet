# -*- coding: utf-8 -*-
# @Description: Main network architecture for GeoMVSNet.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodules import homo_warping, init_inverse_range, schedule_inverse_range, FPN, Reg2d
from models.geometry import GeoFeatureFusion, GeoRegNet2d
from models.filter import frequency_domain_filter


class GeoMVSNet(nn.Module):
    def __init__(self, levels, hypo_plane_num_stages, depth_interal_ratio_stages,
                 feat_base_channel, reg_base_channel, group_cor_dim_stages):
        super(GeoMVSNet, self).__init__()

        self.levels = levels
        self.hypo_plane_num_stages = hypo_plane_num_stages
        self.depth_interal_ratio_stages = depth_interal_ratio_stages

        self.StageNet = StageNet()

        # feature settings
        self.FeatureNet = FPN(base_channels=feat_base_channel)
        self.coarest_separate_flag = True
        if self.coarest_separate_flag:
            self.CoarestFeatureNet = FPN(base_channels=feat_base_channel)
        self.GeoFeatureFusionNet = GeoFeatureFusion(
            convolutional_layer_encoding="z", mask_type="basic", add_origin_feat_flag=True)

        # cost regularization settings
        self.RegNet_stages = nn.ModuleList()
        self.group_cor_dim_stages = group_cor_dim_stages
        self.geo_reg_flag = True
        self.geo_reg_encodings = ['std', 'z', 'z', 'z']  # must use std in idx-0
        for stage_idx in range(self.levels):
            in_dim = group_cor_dim_stages[stage_idx]
            if self.geo_reg_flag:
                self.RegNet_stages.append(GeoRegNet2d(input_channel=in_dim, base_channel=reg_base_channel,
                                                      convolutional_layer_encoding=self.geo_reg_encodings[stage_idx]))
            else:
                self.RegNet_stages.append(Reg2d(input_channel=in_dim, base_channel=reg_base_channel))

        # frequency domain filter settings
        self.curriculum_learning_rho_ratios = [9, 4, 2, 1]

    def get_mask(self, stage_name, ref_lbp_mask, confidence_last):
        lbp_mask_0 = ref_lbp_mask[stage_name][0]
        lbp_mask_1 = ref_lbp_mask[stage_name][1]
        lbp_mask_2 = ref_lbp_mask[stage_name][2]
        lbp_mask_3 = ref_lbp_mask[stage_name][3]
        lbp_mask_4 = ref_lbp_mask[stage_name][4]
        lbp_mask_5 = ref_lbp_mask[stage_name][5]
        lbp_mask_6 = ref_lbp_mask[stage_name][6]
        lbp_mask_7 = ref_lbp_mask[stage_name][7]
        lbp_mask_8 = ref_lbp_mask[stage_name][8]

        if stage_name == "stage1":
            final_mask = lbp_mask_0 == 0
        elif stage_name == "stage2":
            confidence_mask_2 = confidence_last < torch.mean(confidence_last)
            mask_texture = lbp_mask_0 == 0
            final_mask = torch.logical_and(confidence_mask_2, mask_texture)
        elif stage_name == "stage3":
            final_mask = confidence_last > 0
        elif stage_name == "stage4":
            final_mask = confidence_last > 0
        return final_mask

    def forward(self, imgs, proj_matrices, intrinsics_matrices, depth_values, ref_lbp_mask, filename=None):

        features = []
        if self.coarest_separate_flag:
            coarsest_features = []
        for nview_idx in range(len(imgs)):
            img = imgs[nview_idx]
            features.append(self.FeatureNet(img))  # B C H W
            if self.coarest_separate_flag:
                coarsest_features.append(self.CoarestFeatureNet(img))

        # coarse-to-fine
        outputs = {}
        for stage_idx in range(self.levels):
            stage_name = "stage{}".format(stage_idx + 1)
            B, C, H, W = features[0][stage_name].shape
            proj_matrices_stage = proj_matrices[stage_name]
            intrinsics_matrices_stage = intrinsics_matrices[stage_name]

            # @Note features
            if stage_idx == 0:
                confidence_last = None
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

                # reference feature
                features_stage[0] = self.GeoFeatureFusionNet(
                    ref_img_stage, depth_last, confidence_last, depth_values,
                    stage_idx, features_stage[0], intrinsics_matrices_stage
                )
            text_mask = self.get_mask(stage_name, ref_lbp_mask, confidence_last)

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
                regnet=self.RegNet_stages[stage_idx], group_cor_dim=self.group_cor_dim_stages[stage_idx],
                depth_interal_ratio=self.depth_interal_ratio_stages[stage_idx],
                geo_reg_data=geo_reg_data, text_mask=text_mask
            )

            # @Note frequency domain filter
            depth_est = outputs_stage['depth']
            depth_est_filtered = frequency_domain_filter(depth_est,
                                                         rho_ratio=self.curriculum_learning_rho_ratios[stage_idx])
            outputs_stage['depth_filtered'] = depth_est_filtered
            depth_last = depth_est_filtered

            confidence_last = outputs_stage['photometric_confidence']
            prob_volume_last = outputs_stage['prob_volume']

            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


class StageNet(nn.Module):
    def __init__(self, attn_temp=2):
        super(StageNet, self).__init__()
        self.attn_temp = attn_temp

    def forward(self, stage_idx, features, proj_matrices, depth_hypo, regnet,
                group_cor_dim, depth_interal_ratio, geo_reg_data=None, text_mask=None):

        # @Note step1: feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]
        num_views = len(proj_matrices)

        # @Note step2.1: cost aggregation used sim
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        var_volume_sum = ref_volume
        var_volume_sq_sum = ref_volume ** 2
        cor_weight_sum = 1e-8
        cor_feats = 0
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            save_fn = None
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            if stage_idx == 0:
                var_warped_src = warped_src
                var_volume_sum = var_volume_sum + var_warped_src
                var_volume_sq_sum = var_volume_sq_sum + var_warped_src ** 2

            warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)

            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            del warped_src, src_proj, src_fea

            cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
            cor_weight_sum += cor_weight  # B D H W
            cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat

        sim_cost_volume = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        volume_variance = var_volume_sq_sum.div_(num_views).sub_(var_volume_sum.div_(num_views).pow_(2))

        del cor_weight_sum, src_features

        # @Note step3: cost regularization
        if geo_reg_data == {}:
            # basic
            cost_reg = regnet(sim_cost_volume)
        else:
            # probability volume geometry embedding
            cost_reg = regnet(sim_cost_volume, stage_idx, geo_reg_data)
            var_cost_reg = regnet(volume_variance, stage_idx, geo_reg_data)
        del sim_cost_volume, volume_variance
        sim_prob_volume = F.softmax(cost_reg, dim=1)  # B D H W
        var_prob_volume = F.softmax(var_cost_reg, dim=1)  # B D H W
        if stage_idx == 0:
            mask_4d = text_mask.unsqueeze(1).repeat(1, D, 1, 1)
            prob_volume = (1 - mask_4d) * sim_prob_volume + mask_4d * var_prob_volume
        else:
            prob_volume = torch.mul(sim_prob_volume, var_prob_volume)
            prob_volume = F.softmax(prob_volume, dim=1)

        # @Note step4: depth regression
        prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W

        with torch.no_grad():
            # var_based confidence
            var_photometric_confidence = var_prob_volume.max(1)[0]
            var_photometric_confidence = F.interpolate(var_photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear',
                                                   align_corners=True).squeeze(1)
            # sim_based confidence
            sim_photometric_confidence = sim_prob_volume.max(1)[0]
            sim_photometric_confidence = F.interpolate(sim_photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear',
                                                   align_corners=True).squeeze(1)
            # total
            photometric_confidence = prob_volume.max(1)[0]  # B H W
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear',
                                                   align_corners=True).squeeze(1)

        last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
        inverse_min_depth = 1 / depth + depth_interal_ratio * last_depth_itv  # B H W
        inverse_max_depth = 1 / depth - depth_interal_ratio * last_depth_itv  # B H W


        output_stage = {
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            "var_photometric_confidence": var_photometric_confidence,
            "sim_photometric_confidence": sim_photometric_confidence,
            "depth_hypo": depth_hypo,
            "prob_volume": prob_volume,
            "inverse_min_depth": inverse_min_depth,
            "inverse_max_depth": inverse_max_depth,
        }
        return output_stage
