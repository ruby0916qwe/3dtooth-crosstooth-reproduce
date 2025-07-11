import os
import subprocess

def batch_eval():
    """
    对test_gt与test_predict_results中对应的ply文件运行eval_ply_metric.py
    """
    pred_dir = "7.9test_results"
    gt_dir = "7.9test_gt"
    results_dir = "7.9eval_results"
    
    # 创建结果文件夹
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建文件夹: {results_dir}")
    
    # 获取预测文件列表
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_mask.ply')]
    
    print(f"读取到 {len(pred_files)} 个预测文件")
    print("开始批量评估...")
    
    for i, pred_file in enumerate(pred_files, 1):
        # 构建对应的gt文件名
        base_name = pred_file.replace('_mask.ply', '')
        gt_file = f"{base_name}_gt.ply"
        
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        print(f"[{i}/{len(pred_files)}] 评估: {pred_file}")
        
        # 检查gt文件是否存在
        if not os.path.exists(gt_path):
            print(f"  ❌ 找不到对应的gt文件: {gt_file}")
            continue
        
        # 运行评估
        cmd = [
            'python', 'eval_ply_metric.py',
            '--pred', pred_path,
            '--gt', gt_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ 成功: {pred_file}")
            
            # 保存结果到文件
            result_file = os.path.join(results_dir, f"{base_name}_eval_result.txt")
            with open(result_file, 'w') as f:
                f.write(f"预测文件: {pred_file}\n")
                f.write(f"真实标签: {gt_file}\n")
                f.write(f"评估结果:\n{result.stdout}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 失败: {e}")
            print(f"     错误信息: {e.stderr}")
        except KeyboardInterrupt:
            print("\n用户中断，停止处理")
            break

if __name__ == "__main__":
    batch_eval() 