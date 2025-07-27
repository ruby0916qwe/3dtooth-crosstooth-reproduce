import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import glob, re
from models.PTv1.point_transformer_seg import PointTransformerSeg38
from dataset.data import ToothData
from metrics import *
import csv
import pandas as pd 
def find_latest_best_model(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "best_miou_epoch_*.pth")
    model_paths = glob.glob(pattern)
    if not model_paths:
        return None
    # 直接用max带key，返回epoch最大对应的路径
    return max(model_paths, key=lambda fp: int(re.search(r"best_miou_epoch_(\d+).*\.pth", fp).group(1)))

def test_and_evaluate(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    model = PointTransformerSeg38(
        in_channels=6, num_classes=args.num_classes, pretrain=False,
        add_cbl=False, enable_pic_feat=False
    ).to(device)

    checkpoint = torch.load(args.pretrain_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_files = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.endswith('.ply')]
    test_files.sort()
    print(f"Found {len(test_files)} test samples.")
    miou_list = []
    iou_0_list = []

    merged_ious_list = [] 
    per_class_ious_list = []

    with torch.no_grad():
        for ply_path in tqdm(test_files, desc="Testing samples"):
            dataset = ToothData(args, [ply_path], with_label=True, augment=False)
            pointcloud, labels, face_info = dataset[0]  

            pointcloud = pointcloud.unsqueeze(0).to(device)  # (1, N, 6)
            pointcloud = pointcloud.permute(0, 2, 1).contiguous()  # (1, 6, N)

            outputs = model(pointcloud)
            outputs = outputs[0]
            pred_labels = outputs.argmax(dim=1)  # (1, N)
            # pred_labels[pred_labels == 17] = 0
            # pred_labels[pred_labels == 18] = 0

            gt_labels = labels.long().cpu().numpy()
            pred_labels = pred_labels.squeeze(0).cpu().numpy()  

            centers, _ = read_ply_face_center_and_labels(ply_path)
            
            miou, iou_0 = calculate_miou(gt_labels, pred_labels, n_class=17)
            per_class_iou = calculate_per_class_iou(gt_labels, pred_labels, n_class=17)
          
            merged_ious = calculate_merged_ious(gt_labels, pred_labels, eps=True)
            
            # 统计列表
            miou_list.append(miou.item())

            iou_0_list.append(iou_0.item())
            merged_ious_list.append(merged_ious)
            per_class_ious_list.append(per_class_iou)


    merged_iou_counts = {}
    merged_iou_sums = {}
    for sample_dict in merged_ious_list:
        for k, v in sample_dict.items():
            merged_iou_sums[k] = merged_iou_sums.get(k, 0) + v
            merged_iou_counts[k] = merged_iou_counts.get(k, 0) + 1
    merged_iou_means = {k: merged_iou_sums[k] / merged_iou_counts[k] * 100 for k in merged_iou_sums}

    print("\nAverage merged IoU per combination over all samples:")
    
    print("\nMerged pairs出现样本数:")
    for k in sorted(merged_iou_counts.keys()):
        print(f"  {k}: {merged_iou_counts[k]}")

    for k in sorted(merged_iou_means.keys()):
        print(f"  {k}: {merged_iou_means[k] :.2f}")

    miou = np.mean(miou_list) * 100
    print(f"\nMean mIoU over {len(miou_list)} samples: {miou:.2f}")

    iou_0 = np.mean(iou_0_list) * 100
    print(f"\nBackground IoU over {len(iou_0_list)} samples: {iou_0:.2f}")

    # 构造DataFrame数据

    # 先按字母序排序merged_iou_means的key和值，转换为百分比
    merged_keys = sorted(merged_iou_means.keys())
    merged_vals = [merged_iou_means[k] for k in merged_keys]

    data = {
        "ind": [args.pretrain_model_path],
        "mIoU": [miou],
        "Background": [iou_0],
    }

    # 添加 merged iou 各项
    for k, v in zip(merged_keys, merged_vals):
        data[k] = [v]

    df = pd.DataFrame(data)
    df = df.round(2)
    df.to_csv("results_metrics.csv", mode='a', header=False, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="exp/data/test")
    parser.add_argument('--pretrain_model_path', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=17)
    parser.add_argument('--num_points', type=int, default=16000) 
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--version')
    parser.add_argument('--threshold', type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    if args.pretrain_model_path is None:
        checkpoint_dir = os.path.join("exp", args.version, "checkpoints")
        args.pretrain_model_path = find_latest_best_model(checkpoint_dir)
    
    print(f"Using best model:{args.pretrain_model_path}")
    
    test_and_evaluate(args)