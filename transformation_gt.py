import pickle
import numpy as np
from plyfile import PlyData, PlyElement
from utils import label2color_upper, label2color_lower

def label_to_color(label, upper=True):
    if upper:
        color_dict = label2color_upper
    else:
        color_dict = label2color_lower
    color = color_dict.get(int(label), None)
    if color is not None:
        rgb = color[2]
        # 特殊处理：将 (204, 126, 126) 改为 (240, 0, 0)
        if rgb == (204, 126, 126):
            return (240, 0, 0, 255)
        # 特殊处理：将 (255, 234, 157) 改为 (251, 255, 3)
        if rgb == (255, 234, 157):
            return (251, 255, 3, 255)       
        if rgb == (26, 125, 255):
            return (44, 251, 255, 255)
        if rgb == (125, 255, 155):
            return (241, 47, 255, 255)
        if rgb == (241, 47, 255):
            return (125, 255, 155, 255)
        if rgb == (44, 251, 255):
            return (26, 125, 255, 255)
        if rgb == (251, 255, 3):
            return (255, 234, 157, 255)
        return (*rgb, 255)
    else:
        return (125, 125, 125, 255)

def main(pt_path, ply_path, upper=True):
    with open(pt_path, 'rb') as f:
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = pickle.load(f)
    all_vertices = mesh_triangles.reshape(-1, 3)
    vertices_unique, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)
    faces = inverse_indices.reshape(-1, 3)
    vertex_array = np.array(vertices_unique, dtype=np.float32)
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face_dtype = [
        ('vertex_indices', 'i4', (3,)),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1')
    ]
    face_elements = []
    for i, face in enumerate(faces):
        color = label_to_color(labels[i], upper=upper)
        face_elements.append((tuple(face), *color))
    vertex_elements = np.array([tuple(v) for v in vertex_array], dtype=vertex_dtype)
    face_elements = np.array(face_elements, dtype=face_dtype)
    el_verts = PlyElement.describe(vertex_elements, 'vertex')
    el_faces = PlyElement.describe(face_elements, 'face')
    PlyData([el_verts, el_faces], text=True).write(ply_path)
    print(f"PLY 文件已保存：{ply_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("用法: python pt2facecolor_ply.py 输入.pt 输出.ply upper_or_lower")
    else:
        upper = sys.argv[3].lower() == "upper" 
        main(sys.argv[1], sys.argv[2], upper)