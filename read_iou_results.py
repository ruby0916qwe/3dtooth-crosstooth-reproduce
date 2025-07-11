import os
import re

def read_iou_results():
    """
    读取eval_results中每个txt文件的mIoU、bIoU和T1/T9到T7/T15 IOU比值
    统计所有值，不设定阈值
    """
    results_dir = "7.11eval_results"
    
    if not os.path.exists(results_dir):
        print(f"错误: {results_dir} 文件夹不存在")
        return
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"在 {results_dir} 中没有找到.txt文件")
        return
    
    print(f"读取到 {len(txt_files)} 个结果文件")
    print("\n整体指标和对称牙齿对IOU比值统计:")
    print("统计所有值，不设定阈值")
    print("-" * 80)
    
    # 存储所有指标
    all_metrics = {
        'mIoU': [],
        'bIoU': [],
        'T1/T9': [],
        'T2/T10': [],
        'T3/T11': [],
        'T4/T12': [],
        'T5/T13': [],
        'T6/T14': [],
        'T7/T15': []
    }
    
    for txt_file in txt_files:
        file_path = os.path.join(results_dir, txt_file)
        
        try:
            # 尝试多种编码格式
            content = None
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                continue
                
            # 查找mIoU和bIoU
            miou_match = re.search(r'weighted_mIOU:\s*(\d+\.?\d*)', content, re.IGNORECASE)
            biou_match = re.search(r'bIOU\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
            
            if miou_match:
                miou = float(miou_match.group(1))
                all_metrics['mIoU'].append(miou)
            
            if biou_match:
                biou = float(biou_match.group(1))
                all_metrics['bIoU'].append(biou)
            
            
            # 查找各个对称牙齿对的IOU值
            tooth_pairs = ['T1/T9', 'T2/T10', 'T3/T11', 'T4/T12', 'T5/T13', 'T6/T14', 'T7/T15']
            
            for pair in tooth_pairs:
                pattern = rf'{pair} IOU:\s*(\d+\.?\d*)'
                match = re.search(pattern, content, re.IGNORECASE)
                
                if match:
                    ratio = float(match.group(1))
                    all_metrics[pair].append(ratio)  # 统计所有值
        except Exception as e:
            continue
    
    # 统计信息
    print(f"{'指标':<10} {'平均值':<10}")
    print("-" * 80)
    
    # 先显示整体指标（去掉background，只输出平均值）
    for metric in ['mIoU', 'bIoU']:
        values = all_metrics[metric]
        if values:
            avg_value = sum(values) / len(values)
            print(f"{metric:<10} {avg_value:<10.4f}")
        else:
            print(f"{metric:<10} {'N/A':<10}")
    
    print("-" * 80)
    
    # 再显示牙齿对指标（只输出平均值）
    tooth_pairs = ['T1/T9', 'T2/T10', 'T3/T11', 'T4/T12', 'T5/T13', 'T6/T14', 'T7/T15']
    for pair in tooth_pairs:
        values = all_metrics[pair]
        if values:
            avg_value = sum(values) / len(values)
            print(f"{pair:<10} {avg_value:<10.4f}")
        else:
            print(f"{pair:<10} {'N/A':<10}")
    
    # 保存汇总结果
    summary_file = "optimized_threshold_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("整体指标和对称牙齿对IOU比值汇总结果\n")
        f.write("统计所有值，不设定阈值\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'指标':<10} {'平均值':<10} {'最高值':<10} {'最低值':<10} {'保留率':<10}\n")
        f.write("-" * 80 + "\n")
        
        # 保存整体指标
        for metric in ['mIoU', 'bIoU']:
            values = all_metrics[metric]
            if values:
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                f.write(f"{metric:<10} {avg_value:<10.4f} {max_value:<10.4f} {min_value:<10.4f}\n")
            else:
                f.write(f"{metric:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n")
        
        f.write("-" * 80 + "\n")
        
        # 保存牙齿对指标
        for pair in tooth_pairs:
            values = all_metrics[pair]
            if values:
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                retention_rate = len(values) / len(txt_files) * 100
                f.write(f"{pair:<10} {avg_value:<10.4f} {max_value:<10.4f} {min_value:<10.4f} {retention_rate:<10.1f}%\n")
            else:
                f.write(f"{pair:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"总文件数: {len(txt_files)}\n")
    
    print("-" * 80)
    print(f"\n汇总结果已保存到: {summary_file}")

if __name__ == "__main__":
    read_iou_results() 