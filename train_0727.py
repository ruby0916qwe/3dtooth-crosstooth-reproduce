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
from metrics import calculate_miou, compute_boundary_mask, calculate_boundary_miou
import time
from sklearn.neighbors import NearestNeighbors
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='exp/data/train')
    parser.add_argument('--val_dir', type=str, default='exp/data/test')
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=1.0, help='Coefficient for CBLLoss')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation for training')
    parser.add_argument('--no_augment', action='store_false', dest='augment', help='Disable data augmentation for training')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--version', type=str, default=None)
    return parser.parse_args()

def evaluate(model, dataloader, num_classes=17):
    model.eval()
    miou_list = []

    with torch.no_grad():
        for pointcloud, labels, _ in dataloader:
            device = next(model.parameters()).device  # 自动获取模型所在设备
            pointcloud = pointcloud.to(device).permute(0, 2, 1).contiguous()
            labels = labels.to(device)

            outputs = model(pointcloud)
            outputs = outputs[0]

            preds = outputs.argmax(dim=1)  # (B, N)

            batch_size = preds.shape[0]

            for i in range(batch_size):
                sample_pred = preds[i].cpu().numpy()  
                sample_label = labels[i].cpu().numpy() 
                miou, _ = calculate_miou(
                    sample_pred, sample_label, n_class=num_classes)
                miou_list.append(miou.clone().detach())

    model.train()

    miou_tensor = torch.stack(miou_list)
    return miou_tensor.mean().item()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # 训练集文件列表
    train_file_list = glob.glob(os.path.join(args.train_dir, '*.ply'))
    print(f"Found {len(train_file_list)} ply files in {args.train_dir}")

    train_dataset = ToothData(args, train_file_list, with_label=True, augment=args.augment)
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

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_cbl = CBLLoss().to(device)
    # criterion_edge = torch.nn.CrossEntropyLoss() 

    os.makedirs(args.save_dir, exist_ok=True)

    model.train()
    best_models = []
    start_time_all = time.time() 

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        
        for batch_idx, (pointcloud, labels, face_info) in enumerate(train_loader):
            pointcloud = pointcloud.to(device).permute(0, 2, 1).contiguous()  # (B, 6, N)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()

            if model.add_cbl:
                outputs, edge_seg_result, cbl_inputs = model(pointcloud)
            else:
                outputs, edge_seg_result = model(pointcloud)
                cbl_inputs = None

            loss_ce = criterion_ce(outputs, labels)

            if cbl_inputs is not None:
                p1, x1, o1 = cbl_inputs
                labels_flat = labels.view(-1)
                loss_cbl = criterion_cbl([p1, x1, o1], labels_flat)
                loss = loss_ce + args.lam *loss_cbl
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_miou = evaluate(model, train_loader, num_classes=17)
        val_miou = evaluate(model, val_loader, num_classes=17)
        
        # 记录epoch的指标到TensorBoard
        writer.add_scalar('loss', avg_loss, epoch + 1)
        writer.add_scalar('train_mIoU', train_miou, epoch + 1)
        writer.add_scalar('val_mIoU', val_miou, epoch + 1)

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time_all
        epochs_left = args.epochs - (epoch + 1)
        eta_seconds = (elapsed_time / (epoch + 1)) * epochs_left

        # 计算ETA小时、分钟、秒
        eta_h, rem = divmod(int(eta_seconds), 3600)
        eta_m, eta_s = divmod(rem, 60)

        # 计算已用时间小时、分钟、秒
        elapsed_h, rem = divmod(int(elapsed_time), 3600)
        elapsed_m, elapsed_s = divmod(rem, 60)

        elapsed_str = f"{elapsed_h}h {elapsed_m}m {elapsed_s}s" if elapsed_h > 0 else f"{elapsed_m}m {elapsed_s}s"

        eta_str = f"{eta_h}h {eta_m}m {eta_s}s" if eta_h > 0 else f"{eta_m}m {eta_s}s"

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Train mIoU: {train_miou:.4f}, Val mIoU: {val_miou:.4f} "
            f"| Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed_str} | ETA: {eta_str}")
        
        if len(best_models) < 5:
            # 不满5个，直接保存
            save_path = os.path.join(args.save_dir, f"best_miou_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            best_models.append((val_miou, save_path))
            best_models.sort(key=lambda x: x[0], reverse=True)
            print(f"Saved new best model #{len(best_models)}: {save_path} with val mIoU {val_miou:.4f}")
        else:
            best_miou, best_path = best_models[0]
            worst_miou, worst_path = best_models[-1]

            # 只有当val_miou > best_miou才替换最差模型
            if val_miou > best_miou:
                # 删除最差模型文件
                if os.path.exists(worst_path):
                    os.remove(worst_path)

                miou_str = f"{val_miou:.4f}"  # '0.8321'


                miou_str = miou_str.split('.')[1]  # '8321'

                save_path = os.path.join(args.save_dir, f"best_miou_epoch_{epoch+1}_valmiou_{miou_str}.pth")
                torch.save(model.state_dict(), save_path)

                best_models[-1] = (val_miou, save_path)
                best_models.sort(key=lambda x: x[0], reverse=True)
                print(f"Saved new best model #{len(best_models)}: {save_path} with val mIoU {val_miou:.4f}")
        if (epoch + 1) % 20 == 0:
            miou_str = f"{val_miou:.4f}"      # 例如 "0.8321"
            miou_str = miou_str.split('.')[1] # 取小数点后部分 "8321"
            save_path_20 = os.path.join(args.save_dir, f"epoch_{epoch + 1}_valmiou_{miou_str}.pth")
            torch.save(model.state_dict(), save_path_20)
            print(f"Saved checkpoint every 20 epochs: {save_path_20}")

        scheduler.step()

    writer.close()

def main():
    args = get_args()
    version = args.version.strip()

    if args.save_dir is None:
        args.save_dir = os.path.join("exp", version, "checkpoints")
    if args.log_dir is None:
        args.log_dir = os.path.join("exp", version, "runs")

    print(f"Using train_dir: {args.train_dir}")
    print(f"Using val_dir: {args.val_dir}")
    print(f"Using save_dir: {args.save_dir}")
    print(f"Data augmentation enabled: {args.augment}")
    print(f"Using lambda: {args.lam}")
    print(f"Using lr: {args.lr}")
    print(f"Using batch size: {args.batch_size}")

    train(args)

if __name__ == '__main__':
    main()