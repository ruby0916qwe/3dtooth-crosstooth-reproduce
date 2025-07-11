import os
import subprocess
import sys
from pathlib import Path

def batch_transformation_gt():
    """
    使用transformation_gt.py对test文件夹中的每个pt文件进行转换
    生成对应的gt文件并保存到test_gt文件夹
    """
    # 源文件夹和目标文件夹
    source_dir = "test"
    target_dir = "test_gt"
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源文件夹 {source_dir} 不存在")
        return
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建文件夹: {target_dir}")
    
    # 获取所有pt文件
    pt_files = [f for f in os.listdir(source_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print(f"在 {source_dir} 中没有找到.pt文件")
        return
    
    print(f"找到 {len(pt_files)} 个.pt文件，开始批量处理...")
    
    # 统计变量
    success_count = 0
    error_count = 0
    error_files = []
    
    # 处理每个文件
    for i, pt_file in enumerate(pt_files, 1):
        # 提取文件名（去掉data_前缀和.pt后缀）
        base_name = pt_file.replace('data_', '').replace('.pt', '')
        
        # 构建输出文件名
        output_file = f"{base_name}_gt.ply"
        output_path = os.path.join(target_dir, output_file)
        
        # 构建完整的pt文件路径
        pt_path = os.path.join(source_dir, pt_file)
        
        print(f"[{i}/{len(pt_files)}] 处理: {pt_file} -> {output_file}")
        
        try:
            # 运行transformation_gt.py
            # 格式: python transformation_gt.py input.pt output.ply upper/lower
            # 由于所有文件都是upper，所以传入"upper"参数
            result = subprocess.run([
                sys.executable, 'transformation_gt.py', 
                pt_path, output_path, "lower"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"  ✅ 成功: {output_file}")
                success_count += 1
            else:
                print(f"  ❌ 失败: {pt_file}")
                print(f"     错误信息: {result.stderr}")
                error_count += 1
                error_files.append(pt_file)
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时: {pt_file}")
            error_count += 1
            error_files.append(pt_file)
        except Exception as e:
            print(f"  💥 异常: {pt_file} - {str(e)}")
            error_count += 1
            error_files.append(pt_file)
    
    # 输出最终统计结果
    print(f"\n=== 批量处理完成 ===")
    print(f"总文件数: {len(pt_files)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    print(f"成功率: {success_count/len(pt_files)*100:.1f}%")
    
    if error_files:
        print(f"\n失败的文件:")
        for file in error_files:
            print(f"  - {file}")
    
    if success_count > 0:
        print(f"\n成功生成的文件保存在: {target_dir}/")

if __name__ == "__main__":
    batch_transformation_gt() 