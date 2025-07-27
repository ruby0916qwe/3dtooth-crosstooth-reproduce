import torch
import numpy as np
from .provider import rotate_perturbation_point_cloud_with_normal, shift_point_cloud


class PointcloudToTensor(object):
    def __call__(self, data):
        points, face_info = data
        return torch.from_numpy(points).float(), face_info


class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid # 平移至中心点
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 将所有点坐标缩放到单位球半径内（最大距离为1）
        pc = pc / m
        return pc

    def __call__(self, data):
        points, face_info = data
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return (torch.from_numpy(pc).float(), face_info)

class PointcloudRandomRotate(object):
    """ 使用 rotate_perturbation_point_cloud_with_normal 实现带法向量点云的小角度随机旋转 """
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data):
        points, face_info = data

        points_np = points.cpu().numpy()
        if points_np.ndim == 2:
            points_np = points_np[None, ...]  # 增加batch维度

        rotated_np = rotate_perturbation_point_cloud_with_normal(
            points_np, self.angle_sigma, self.angle_clip)

        if rotated_np.shape[0] == 1:
            rotated_np = rotated_np[0]

        rotated_tensor = torch.from_numpy(rotated_np).float()
        return rotated_tensor, face_info

class PointcloudRandomShift(object):
    """ 使用 shift_point_cloud 实现随机平移 """
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        points, face_info = data
        points_np = points.numpy()

        if points_np.ndim == 2:
            points_np = points_np[None, ...]

        shifted = shift_point_cloud(points_np, self.shift_range)

        if shifted.shape[0] == 1:
            shifted = shifted[0]

        return torch.from_numpy(shifted).float(), face_info


class PointcloudSample(object):
    def __init__(self, total=16000, sample=16000):
        self.total = total
        self.sample = sample

    def __call__(self, data):
        points, face_info = data
        sample = np.random.permutation(self.total)[:self.sample]
        return (points[sample], face_info[sample])