import os
import os.path as osp
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import argparse


class Gaussian:
    def __init__(self, gaussian):
        self._means = gaussian["_means"].cpu().numpy()
        self._features_dc = gaussian["_features_dc"].cpu().numpy()
        self._features_rest = gaussian["_features_rest"].cpu().numpy()
        self._opacities = gaussian["_opacities"].cpu().numpy()
        self._scales = gaussian["_scales"].cpu().numpy()
        self._quats = gaussian["_quats"].cpu().numpy()


def export_ply(pth_path, road_sh_num, road_only, out_path):
    data = torch.load(pth_path)

    if road_sh_num == 0:
        num_f_rest = 0
    elif road_sh_num == 1:
        num_f_rest = 9
    elif road_sh_num == 2:
        num_f_rest = 24
    elif road_sh_num == 3:
        num_f_rest = 45
    if road_only:
        gaussian_road = Gaussian(data["models"]["RoadNodes"])
        xyz = gaussian_road._means
        normals = np.zeros_like(xyz)
        f_dc = gaussian_road._features_dc
        f_dc = f_dc.reshape((f_dc.shape[0], -1))
        f_rest = gaussian_road._features_rest
        f_rest = f_rest.reshape((f_rest.shape[0], -1))
        opacities = gaussian_road._opacities
        scale = gaussian_road._scales
        rotation = gaussian_road._quats
    else:
        gaussian = Gaussian(data["models"]["Background"])
        gaussian_road = Gaussian(data["models"]["RoadNodes"])
        xyz = np.concatenate([gaussian._means, gaussian_road._means], axis=0)
        normals = np.zeros_like(xyz)
        # 合并 DC 特征
        f_dc = np.concatenate([gaussian._features_dc, gaussian_road._features_dc], axis=0)
        f_dc = f_dc.reshape((f_dc.shape[0], -1))
        # 合并 Rest 特征 - RoadNodes 需要补零
        if gaussian_road._features_rest is None:
            # RoadNodes 没有 Rest 特征，创建全零数组
            # 3阶SH的Rest部分有15个系数，每个系数有3个颜色通道
            road_f_rest = np.zeros((gaussian_road._means.shape[0], 15, 3))
        else:
            road_f_rest = gaussian_road._features_rest
        # 确保两个数组的形状一致
        if gaussian._features_rest.shape[1] != road_f_rest.shape[1]:
            # 如果维度不匹配，需要调整RoadNodes的维度
            # 这里假设RoadNodes是0阶SH，需要扩展到3阶SH的维度
            current_rest_dim = road_f_rest.shape[1]
            target_rest_dim = gaussian._features_rest.shape[1]
            if current_rest_dim < target_rest_dim:
                # 补零到目标维度
                padding = np.zeros((road_f_rest.shape[0], target_rest_dim - current_rest_dim, 3))
                road_f_rest = np.concatenate([road_f_rest, padding], axis=1)
            elif current_rest_dim > target_rest_dim:
                # 截断到目标维度（通常不会发生）
                road_f_rest = road_f_rest[:, :target_rest_dim, :]
        f_rest = np.concatenate([gaussian._features_rest, road_f_rest], axis=0)
        f_rest = f_rest.reshape((f_rest.shape[0], -1))
        opacities = np.concatenate([gaussian._opacities, gaussian_road._opacities], axis=0)
        scale = np.concatenate([gaussian._scales, gaussian_road._scales], axis=0)
        rotation = np.concatenate([gaussian._quats, gaussian_road._quats], axis=0)


    def construct_list_of_attributes(gaussian, num_f_rest):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(3):
            l.append("f_dc_{}".format(i))
        for i in range(num_f_rest):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(gaussian._scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(gaussian._quats.shape[1]):
            l.append("rot_{}".format(i))
        return l
        
    num_f_rest = f_rest.shape[1]
    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(gaussian_road, num_f_rest)
    ]
    attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(attribute_list, axis=1)
    # do not save 'features_extra' for ply
    # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export background gaussians to ply file")

    # misc
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--road_sh_num", type=int, default=1)
    parser.add_argument("--road_only", type=bool, default=False)
    args = parser.parse_args()

    run_dir = os.path.dirname(args.model)

    save_dir = osp.join(run_dir, "point_clouds")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "background_road.ply")

    export_ply(args.model, args.road_sh_num, args.road_only, save_path)
