import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import vedo
from dataset import data_util
from utils import color2label

class ToothData(Dataset):
    def __init__(self, args, file_list, with_label=True):
        """
        :param with_label: 是否返回face_labels。训练时True，推理时False。
        """
        self.args = args
        self.file_list = file_list
        self.with_label = with_label
        self.point_transform = transforms.Compose(
            [
                data_util.PointcloudToTensor(),
                data_util.PointcloudNormalize(radius=1),
                # data_util.PointcloudSample(total=args.num_points, sample=args.sample_points) # 可选
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        mesh = vedo.load(item)
        cell_normals = np.array(mesh.normals(cells=True))
        point_coords = np.array(mesh.points())
        face_info = np.array(mesh.cells())
        cell_coords = np.array([[
            (point_coords[i0][0] + point_coords[i1][0] + point_coords[i2][0]) / 3,
            (point_coords[i0][1] + point_coords[i1][1] + point_coords[i2][1]) / 3,
            (point_coords[i0][2] + point_coords[i1][2] + point_coords[i2][2]) / 3,
        ] for i0, i1, i2 in face_info])

        pointcloud = np.concatenate((cell_coords, cell_normals), axis=1)

        if self.with_label:
            face_colors_rgba = np.array(mesh.cellcolors)
            face_colors = face_colors_rgba[:, :3].astype(np.uint8)
            face_colors_tuples = [tuple(color) for color in face_colors]
            face_labels = np.array([color2label[c][2] for c in face_colors_tuples], dtype=np.int64)

        # padding
        if pointcloud.shape[0] < self.args.num_points:
            pad_len = self.args.num_points - pointcloud.shape[0]
            padding = np.zeros((pad_len, pointcloud.shape[1]))
            pointcloud = np.concatenate((pointcloud, padding), axis=0)

            face_info = np.concatenate((face_info, np.zeros((pad_len, 3))), axis=0)

            if self.with_label:
                face_labels = np.concatenate((face_labels, np.zeros(pad_len, dtype=np.int64)), axis=0)

        if self.with_label:
            permute = np.random.permutation(self.args.num_points)
            pointcloud = pointcloud[permute]
            face_info = face_info[permute]
            face_labels = face_labels[permute]
        else:
            pass

        pointcloud, face_info = self.point_transform([pointcloud, face_info])

        if self.with_label:
            return pointcloud.float(), torch.from_numpy(face_labels).long(), face_info.astype(np.int32)
        else:
            return pointcloud.to(torch.float), point_coords, face_info

    def get_by_name(self, name):
        idx = self.file_list.index(name)
        return self.__getitem__(idx)