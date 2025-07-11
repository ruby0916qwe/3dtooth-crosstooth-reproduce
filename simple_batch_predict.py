import os
import subprocess

def simple_batch_predict():
    """
    对test_for_predict中的每个_upper_for_predict.ply文件运行predict.py
    """
    source_dir = "test_for_predict"
    target_dir = "test_predict_results"
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建文件夹: {target_dir}")
    
    # 获取所有ply文件
    ply_files = [f for f in os.listdir(source_dir) if f.endswith('_for_predict.ply')]
    
    print(f"读取到 {len(ply_files)} 个文件")
    print("文件列表:")
    for i, file in enumerate(ply_files, 1):
        print(f"  {i}. {file}")
    
    print(f"\n开始批量预测...")
    
    for i, ply_file in enumerate(ply_files, 1):
        # 构建输出文件名
        output_file = ply_file.replace('_for_predict.ply', '_mask.ply')
        output_path = os.path.join(target_dir, output_file)
        
        print(f"[{i}/{len(ply_files)}] 处理: {ply_file}")
        
        # 运行predict.py
        cmd = [
            'python', 'predict.py',
            '--case', os.path.join(source_dir, ply_file),
            '--save_path', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✅ 成功: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 失败: {e}")
        except KeyboardInterrupt:
            print("\n用户中断，停止处理")
            break

if __name__ == "__main__":
    simple_batch_predict() 