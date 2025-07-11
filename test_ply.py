from plyfile import PlyData
import numpy as np

# 检查带标签的PLY文件
print("检查 YBSESUN6_upper_mask.ply:")
try:
    ply = PlyData.read('YBSESUN6_upper_mask.ply')
    print("Face properties:", ply['face'].dtype)
    if len(ply['face']) > 0:
        print("Sample face data:", ply['face'][0])
        print("Number of faces:", len(ply['face']))
    else:
        print("No faces found")
except Exception as e:
    print("Error:", e)

print("\n检查 YBSESUN6_upper.ply:")
try:
    ply = PlyData.read('YBSESUN6_upper.ply')
    print("Face properties:", ply['face'].dtype)
    if len(ply['face']) > 0:
        print("Sample face data:", ply['face'][0])
        print("Number of faces:", len(ply['face']))
    else:
        print("No faces found")
except Exception as e:
    print("Error:", e) 