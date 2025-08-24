from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize  # 引入Cython转换工具
import numpy as np
import os
import sys

# # 定义扩展模块
# extensions = [
#     Extension(
#         name="pyfqmr.Simplify",
#         sources=["pyfqmr_src/Simplify.pyx"],  # 直接指定pyx文件，Cython会自动转换为cpp
#         include_dirs=[
#             np.get_include(),
#             os.path.join(os.path.dirname(__file__), "pyfqmr")  # 确保能找到Simplify.h
#         ],
#         language="c++",  # 指定为C++代码
#         # 添加C++11标准支持（解决之前的语法兼容问题）
#         extra_compile_args=["/std:c++11"] if sys.platform.startswith('win32') else ["-std=c++11"]
#     )
# ]

# setup(
#     name="pyfqmr",  # 包名称
#     version="0.8",  # 版本号
#     packages=["pyfqmr_src"],  # 包含的Python包
#     # 使用cythonize处理扩展模块，自动将pyx转换为cpp
#     ext_modules=cythonize(extensions, language_level=3, force=True)
# )

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


# 1. 替换为你的libigl实际路径（确保正确）
LIBIGL_PATH = "D:/lib"  # 例如："D:/libigl"（需包含include和external/eigen）
eigen_path = 'D:/eigen'

# 2. 定义扩展模块
ext_modules = [
    Extension(
        name="pyfqmr.Simplify",  # 模块名
        sources=["pyfqmr_src/Simplify.pyx"],  # 源文件路径（你的实际目录）
        include_dirs=[
            np.get_include(),  # numpy头文件
            f"{LIBIGL_PATH}/include",  # libigl头文件（关键）
            f"{eigen_path}",  # 启用Eigen路径（之前注释掉了，必须添加）
            os.path.join(os.path.dirname(__file__), "pyfqmr_src"),  # 你的代码目录（替换pyfqmr为实际目录）
        ],
        language="c++",
         extra_compile_args=["-std:c++11"] if sys.platform.startswith('win32') else ["-std=c++11"],
    )
]

setup(
    name="pyfqmr",
    version="0.8",
    ext_modules=cythonize(ext_modules),
    # 3. 修正packages参数（使用实际存在的目录pyfqmr_src）
    packages=["pyfqmr_src"],
    # 4. 确保包目录被正确识别（如果pyfqmr_src下没有__init__.py）
    package_dir={"pyfqmr_src": "pyfqmr_src"},
)
    



