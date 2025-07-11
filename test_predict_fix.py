"""
测试predict.py修复的脚本

这个脚本用于测试修复后的predict.py是否能正常工作
"""

import os
import subprocess
import sys

def test_single_prediction():
    """
    测试单个文件的预测
    """
    print("=== 测试predict.py修复 ===")
    
    # 检查必要文件
    if not os.path.exists("test_for_predict"):
        print("❌ test_for_predict文件夹不存在")
        return False
    
    ply_files = [f for f in os.listdir("test_for_predict") if f.endswith('.ply')]
    if not ply_files:
        print("❌ 没有找到ply文件")
        return False
    
    # 选择第一个文件进行测试
    test_file = ply_files[0]
    base_name = test_file.replace('_for_predict.ply', '')
    output_file = f"{base_name}_test_mask.ply"
    
    print(f"测试文件: {test_file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 检查CUDA可用性
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用性: {'是' if cuda_available else '否'}")
        
        # 运行预测
        cmd = [
            sys.executable, 'predict.py',
            '--case', os.path.join("test_for_predict", test_file),
            '--save_path', output_file,
            '--pretrain_model_path', 'models/PTv1/point_best_model.pth',
            '--num_points', '16000',
            '--sample_points', '16000'
        ]
        
        if cuda_available:
            cmd.extend(['--no_cuda', 'False'])
        else:
            cmd.extend(['--no_cuda', 'True'])
        
        print(f"运行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ 测试成功!")
            if os.path.exists(output_file):
                print(f"✅ 输出文件已生成: {output_file}")
                # 清理测试文件
                os.remove(output_file)
                print("✅ 测试文件已清理")
            return True
        else:
            print("❌ 测试失败!")
            print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时!")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_single_prediction()
    if success:
        print("\n🎉 predict.py修复成功，可以运行批量预测了!")
        print("运行命令: python batch_predict.py")
    else:
        print("\n💥 predict.py仍有问题，需要进一步调试") 