import os
import glob
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.PTv1.point_transformer_seg import PointTransformerSeg38
from dataset.data import ToothData  
from loss.cbl import CBLLoss 
from metrics import calculate_miou, compute_boundary_mask
import time
from sklearn.neighbors import NearestNeighbors
from utils import cal_metric

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="default", help="版本名，用于设置保存路径")
    parser.add_argument('--train_dir', type=str, default="b/data/train")
    parser.add_argument('--val_dir', type=str, default="b/data/test")
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=50.0, help="边界正样本loss权重")
    # 不再默认设置 save_dir 和 log_dir，由 version 控制
    return parser.parse_args()

def evaluate(model, dataloader, args, num_classes=2):
    model.eval()
    miou_list = []

    with torch.no_grad():
        for pointcloud, labels, _ in dataloader:
            device = next(model.parameters()).device
            pointcloud = pointcloud.to(device).permute(0, 2, 1).contiguous()  # (B, 6, N)
            labels = labels.to(device)  # (B, N)

            _, edge_seg_result, _ = model(pointcloud)  # edge_seg_result: (B, 2, N)
            batch_size = edge_seg_result.shape[0]
            xyz_batch = pointcloud[:, :3, :].permute(0, 2, 1).cpu().numpy()  # (B, N, 3)

            for i in range(batch_size):
                sample_pred_logits = edge_seg_result[i].unsqueeze(0)  # (1, 2, N)
                face_centers = xyz_batch[i]                            # (N, 3)
                label_np = labels[i].cpu().numpy()                     # (N,)

                boundary_mask = compute_boundary_mask(face_centers, label_np, threshold=args.threshold, k=8)  # bool array (N,)
                sample_label = torch.tensor(boundary_mask.astype(np.int64), device=device)  # (N,) long tensor

                TP, FP, TN, FN, IOU, pred_classes = cal_metric(
                    sample_label, sample_pred_logits)

                miou_list.append(torch.tensor(IOU))

    model.train()

    miou_tensor = torch.stack(miou_list)
    return miou_tensor.mean().item()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 根据 version 参数，动态创建路径
    args.save_dir = os.path.join("b", args.version, "checkpoints")
    args.log_dir = os.path.join("b", args.version, "runs")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    train_file_list = glob.glob(os.path.join(args.train_dir, '*.ply'))
    print(f"Found {len(train_file_list)} ply files in {args.train_dir}")

    train_dataset = ToothData(args, train_file_list, with_label=True, augment=True)
    print(f"Train dataset size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_file_list = glob.glob(os.path.join(args.val_dir, '*.ply'))
    print(f"Found {len(val_file_list)} ply files in {args.val_dir}")

    val_dataset = ToothData(args, val_file_list, with_label=True, augment=False)
    print(f"Val dataset size: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = PointTransformerSeg38(
        in_channels=6, num_classes=17, pretrain=False, enable_pic_feat=False, add_cbl=True 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion_edge = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    best_models = []
    start_time_all = time.time() 

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0

        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
        total_count = 0
        
        for batch_idx, (pointcloud, labels, face_info) in enumerate(train_loader):
            pointcloud = pointcloud.to(device).permute(0, 2, 1).contiguous()  # (B, 6, N)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()

            if model.add_cbl:
                outputs, edge_seg_result, cbl_inputs = model(pointcloud)
            else:
                outputs, edge_seg_result = model(pointcloud)
                cbl_inputs = None

            B, N = labels.shape
            xyz = pointcloud[:, :3, :].permute(0, 2, 1).cpu().numpy()  # (B, N, 3)
            batch_boundary_labels = []

            for b in range(B):
                xyz_b = xyz[b]  # (N, 3)
                label_np = labels[b].cpu().numpy()  # (N,)
                
                boundary_mask = compute_boundary_mask(xyz_b, label_np, threshold=args.threshold, k=8)  # bool (N,)
                boundary_mask_tensor = torch.tensor(boundary_mask.astype(np.float32), device=device)  # (N,) float
                
                batch_boundary_labels.append(boundary_mask_tensor)

            boundary_labels_tensor = torch.stack(batch_boundary_labels)  # (B, N) float

            edge_logits = edge_seg_result[:, 1, :]  

            loss = criterion_edge(edge_logits, boundary_labels_tensor)

            with torch.no_grad():
                for b in range(B):
                    sample_label = boundary_labels_tensor[b]   # (N,)
                    sample_pred_logits = edge_seg_result[b].unsqueeze(0)  # (1, 2, N)
                    TP, FP, TN, FN, IOU, pred_classes = cal_metric(sample_label, sample_pred_logits)
                    total_TP += TP
                    total_FP += FP
                    total_TN += TN
                    total_FN += FN
                    total_count += TP + FP + TN + FN

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_miou = evaluate(model, train_loader, args, num_classes=2)
        val_miou = evaluate(model, val_loader, args, num_classes=2)
        
        writer.add_scalar('Train/Avg_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Train/mIoU', train_miou, epoch + 1)
        writer.add_scalar('Val/mIoU', val_miou, epoch + 1)

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time_all
        epochs_left = args.epochs - (epoch + 1)
        eta_seconds = (elapsed_time / (epoch + 1)) * epochs_left

        eta_h, rem = divmod(int(eta_seconds), 3600)
        eta_m, eta_s = divmod(rem, 60)
        elapsed_h, rem = divmod(int(elapsed_time), 3600)
        elapsed_m, elapsed_s = divmod(rem, 60)

        elapsed_str = f"{elapsed_h}h {elapsed_m}m {elapsed_s}s" if elapsed_h > 0 else f"{elapsed_m}m {elapsed_s}s"
        eta_str = f"{eta_h}h {eta_m}m {eta_s}s" if eta_h > 0 else f"{eta_m}m {eta_s}s"

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Train mIoU: {train_miou:.4f}, Val mIoU: {val_miou:.4f} "
              f"| Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed_str} | ETA: {eta_str}")

        if (epoch + 1) % 5 == 0:
            if total_count > 0:
                tp_ratio = total_TP / total_count
                fp_ratio = total_FP / total_count
                tn_ratio = total_TN / total_count
                fn_ratio = total_FN / total_count

                precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
                recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0

                print(f"Epoch [{epoch+1}] Ratios -> TP: {tp_ratio:.4f}, FP: {fp_ratio:.4f}, TN: {tn_ratio:.4f}, FN: {fn_ratio:.4f} "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            else:
                print(f"Epoch [{epoch+1}] No samples for ratio calculation.")

        if len(best_models) < 5:
            save_path = os.path.join(args.save_dir, f"best_miou_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            best_models.append((val_miou, save_path))
            best_models.sort(key=lambda x: x[0], reverse=True)
            print(f"Saved new best model #{len(best_models)}: {save_path} with val mIoU {val_miou:.4f}")
        else:
            best_miou, best_path = best_models[0]
            worst_miou, worst_path = best_models[-1]

            if val_miou > worst_miou:
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"Removed worst model: {worst_path}")

                save_path = os.path.join(args.save_dir, f"best_miou_epoch_{epoch+1}_valmiou_{val_miou:.4f}.pth")
                torch.save(model.state_dict(), save_path)

                best_models[-1] = (val_miou, save_path)
                best_models.sort(key=lambda x: x[0], reverse=True)
                print(f"Saved new best model #{len(best_models)}: {save_path} with val mIoU {val_miou:.4f}")

        scheduler.step()

        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f"point_transformer_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    writer.close()

def main():
    args = get_args()

    print(f"Using train_dir: {args.train_dir}")
    print(f"Using val_dir: {args.val_dir}")
    print(f"Using version: {args.version}")
    print(f"Checkpoints will be saved to: b/{args.version}/checkpoints")
    print(f"TensorBoard logs will be saved to: b/{args.version}/runs")

    train(args)

if __name__ == '__main__':
    main()