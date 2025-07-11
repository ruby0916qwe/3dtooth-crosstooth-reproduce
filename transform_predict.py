import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import sys

def main(pt_path, ply_path):
    # 读取pt文件
    with open(pt_path, 'rb') as f:
        mesh_faces, mesh_triangles, *_ = pickle.load(f)

    # 还原唯一顶点和面索引
    all_vertices = mesh_triangles.reshape(-1, 3)
    vertices_unique, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)
    faces = inverse_indices.reshape(-1, 3)
    vertex_array = np.array(vertices_unique, dtype=np.float32)

    # 所有点的颜色为(127,127,127,255)
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')]
    vertex_elements = np.array(
        [(*v, 127, 127, 127, 255) for v in vertex_array],
        dtype=vertex_dtype
    )

    # 面只写索引，不写颜色
    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_elements = np.array(
        [(tuple(face),) for face in faces],
        dtype=face_dtype
    )

    el_verts = PlyElement.describe(vertex_elements, 'vertex')
    el_faces = PlyElement.describe(face_elements, 'face')
    PlyData([el_verts, el_faces], text=True).write(ply_path)
    print(f"预测用 PLY 文件已保存：{ply_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python pt2predict_ply.py 输入.pt 输出.ply")
    else:
        main(sys.argv[1], sys.argv[2])
