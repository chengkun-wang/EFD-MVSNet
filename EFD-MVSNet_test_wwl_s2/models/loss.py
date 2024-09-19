# -*- coding: utf-8 -*-
# @Description: Loss Functions (Sec 3.4 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import torch


def geomvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    depth_values = kwargs.get("depth_values")
    depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_loss1 = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    pw_loss_stages = []
    dds_loss_stages = []
    pp_loss_stages = []
    dr_loss_stages = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        prob_volume = stage_inputs['prob_volume']
        depth_value = stage_inputs['depth_hypo']
        text_mask = stage_inputs['text_mask']
        sim_confidence = stage_inputs['sim_photometric_confidence']
        var_confidence = stage_inputs['var_photometric_confidence']
        depth_filter = stage_inputs['depth_filtered']
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        if stage_idx != 0:
            refined_depth = stage_inputs["depth_refine"]
            depth_refine_loss = depth_filter_loss(refined_depth.squeeze(1), depth_gt, mask)
            dr_loss_stages.append(depth_refine_loss)
            total_loss1 = total_loss1 + depth_refine_loss

        # pw loss
        pw_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_value)
        pw_loss_stages.append(pw_loss)

        # p1p2 loss
        pp_loss = depth_confidence_loss(sim_confidence, var_confidence, mask)
        pp_loss_stages.append(pp_loss)

        # dds loss
        dds_loss = depth_distribution_similarity_loss(depth_filter, depth_gt, mask, depth_min, depth_max)
        dds_loss_stages.append(dds_loss)

        # total loss
        lam1, lam2, lam3 = 0.6, 0.2, 0.2
        total_loss = total_loss + stage_lw[stage_idx] * (lam1 * pw_loss + lam2 * pp_loss + lam3 * dds_loss)

    total_loss = total_loss + total_loss1
    depth_pred = stage_inputs['depth']
    refined_depth = stage_inputs["depth_refine"]
    depth_gt = depth_gt_ms[stage_key]
    epe_depth_pre = cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max)
    epe_depth_refine = cal_metrics(refined_depth.squeeze(1), depth_gt, mask, depth_min, depth_max)

    return total_loss, epe_depth_pre, epe_depth_refine, pw_loss_stages, dds_loss_stages, pp_loss_stages,dr_loss_stages


def pixel_wise_loss(prob_volume, depth_gt, mask, depth_value):
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-12

    shape = depth_gt.shape

    depth_num = depth_value.shape[1]
    depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1,
                                                                                                           gt_index_image,
                                                                                                           1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-12), dim=1).squeeze(1)
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    pw_loss = masked_cross_entropy
    return pw_loss


def depth_confidence_loss(sim_confidence, var_confidence, mask):
    pp = torch.mul(sim_confidence, var_confidence)
    pp_loss = -1 * (1 - pp) * torch.log(pp)
    pp_loss = torch.mul(pp_loss,mask)
    pp_loss = torch.sum(pp_loss, dim=[1,2])
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-12
    pp_loss = torch.mean(pp_loss/valid_pixel_num)

    return pp_loss


def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err1 = (abs_err <= 1).float().mean() * 100
    err3 = (abs_err <= 3).float().mean() * 100

    return epe  # err1, err3


def depth_filter_loss(depth_refine, depth_gt, mask):
    assert depth_refine.dim() == depth_gt.dim(), "inconsistent dimensions"
    l1_loss = torch.abs(depth_refine-depth_gt)
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-12
    l1_loss = torch.mul(l1_loss,mask)
    l1_loss = torch.sum(l1_loss, dim=[1, 2])
    l1_loss = torch.mean(l1_loss/ valid_pixel_num)
    return l1_loss


def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max):
    depth_norm = depth * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    M_bins = 48
    kl_min = torch.min(torch.min(depth_gt), depth.mean() - 3. * depth.std())
    kl_max = torch.max(torch.max(depth_gt), depth.mean() + 3. * depth.std())
    bins = torch.linspace(kl_min, kl_max, steps=M_bins)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i + 1])
        merged_mask = mask & bin_mask

        if merged_mask.sum() > 0:
            p = depth_norm[merged_mask]
            q = depth_gt_norm[merged_mask]
            kl_div = torch.nn.functional.kl_div(torch.log(p) - torch.log(q), p, reduction='batchmean')
            kl_div = torch.log(kl_div)
            kl_divs.append(kl_div)

    dds_loss = sum(kl_divs)
    return dds_loss
