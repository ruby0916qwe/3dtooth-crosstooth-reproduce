import os
import subprocess
import sys
from pathlib import Path

def batch_predict():
    """
    使用test_for_predict中的数据来运行predict.py
    对每个输出结果生成对应的mask文件
    """
    # 源文件夹和目标文件夹
    source_dir = "test_for_predict"
    target_dir = "test_predict_results"
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源文件夹 {source_dir} 不存在")
        return
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建文件夹: {target_dir}")
    
    # 获取所有ply文件
    ply_files = [f for f in os.listdir(source_dir) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"在 {source_dir} 中没有找到.ply文件")
        return
    
    print(f"找到 {len(ply_files)} 个.ply文件，开始批量预测...")
    
    # 检查CUDA可用性
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用性: {'是' if cuda_available else '否'}")
        if cuda_available:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("警告: 无法导入torch，将使用CPU模式")
        cuda_available = False
    
    # 统计变量
    success_count = 0
    error_count = 0
    error_files = []
    
    # 处理每个文件
    for i, ply_file in enumerate(ply_files, 1):
        # 提取文件名（去掉_for_predict.ply后缀）
        base_name = ply_file.replace('_for_predict.ply', '')
        
        # 构建输出文件名
        output_file = f"{base_name}_mask.ply"
        output_path = os.path.join(target_dir, output_file)
        
        # 构建完整的输入文件路径
        input_path = os.path.join(source_dir, ply_file)
        
        print(f"[{i}/{len(ply_files)}] 预测: {ply_file} -> {output_file}")
        
        try:
            # 运行predict.py
            # 格式: python predict.py --case input.ply --save_path output.ply
            cmd = [
                sys.executable, 'predict.py',
                '--case', input_path,
                '--save_path', output_path,
                '--pretrain_model_path', 'models/PTv1/point_best_model.pth',
                '--num_points', '16000',
                '--sample_points', '16000'
            ]
            
            # 根据CUDA可用性设置参数
            if cuda_available:
                cmd.extend(['--no_cuda', 'False'])
            else:
                cmd.extend(['--no_cuda', 'True'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 增加超时时间到5分钟
            
            if result.returncode == 0:
                print(f"  ✅ 成功: {output_file}")
                success_count += 1
            else:
                print(f"  ❌ 失败: {ply_file}")
                print(f"     错误信息: {result.stderr}")
                error_count += 1
                error_files.append(ply_file)
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时: {ply_file}")
            error_count += 1
            error_files.append(ply_file)
        except Exception as e:
            print(f"  💥 异常: {ply_file} - {str(e)}")
            error_count += 1
            error_files.append(ply_file)
    
    # 输出最终统计结果
    print(f"\n=== 批量预测完成 ===")
    print(f"总文件数: {len(ply_files)}")
    print(f"成功预测: {success_count}")
    print(f"预测失败: {error_count}")
    print(f"成功率: {success_count/len(ply_files)*100:.1f}%")
    
    if error_files:
        print(f"\n失败的文件:")
        for file in error_files:
            print(f"  - {file}")
    
    if success_count > 0:
        print(f"\n成功生成的文件保存在: {target_dir}/")

if __name__ == "__main__":
    batch_predict() 