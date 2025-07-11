import os
from pathlib import Path

def count_files_in_test():
    """
    统计test文件夹中的文件数量
    """
    test_dir = "test"
    
    # 检查test文件夹是否存在
    if not os.path.exists(test_dir):
        print(f"错误: {test_dir} 文件夹不存在")
        return
    
    # 获取test文件夹中的所有文件
    try:
        files = os.listdir(test_dir)
        file_count = len(files)
        
        print(f"=== {test_dir} 文件夹统计 ===")
        print(f"文件总数: {file_count}")
        
        # 按文件类型统计
        file_types = {}
        for file in files:
            if '.' in file:
                ext = file.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['无扩展名'] = file_types.get('无扩展名', 0) + 1
        
        if file_types:
            print(f"\n按文件类型统计:")
            for ext, count in sorted(file_types.items()):
                print(f"  .{ext}: {count} 个文件")
        
        # 显示前10个文件名作为示例
        if files:
            print(f"\n前10个文件示例:")
            for i, file in enumerate(files[:10]):
                print(f"  {i+1}. {file}")
            
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
        
    except Exception as e:
        print(f"读取文件夹时出错: {e}")

if __name__ == "__main__":
    count_files_in_test() 