(请先下载拓展`Markdown Preview Enhanced`以便阅览.md)
## Data
`data`：新增文件夹，存放数据
```
data
    |process  
        | train_downsampled 
    | test_lower 
    | test_upper
    | train_lower
    | train_upper
    | downsample.py

```
`test_lower,test_upper,train_lower,train_upper`:全部的teeth3ds对应的ply
`train_downsampled`:部分downsample后数据
## Usage

`utils.py`, `dataset/data.py`: 有改动

`process.py`：把Downloads路径下原teeth3ds数据集(obj,json)转为ply格式（`test_lower,test_upper,train_lower,train_upper`）
运行前，修改99-100行的output文件夹路径。可以去掉test_dir部分(不上色的ply), 这里的test可能理解为inference会更好

`downsample.py`: qem downsample
运行前，修改32-33行的文件夹路径
```commandline
cd data 
python downsample.py 
```

predict:
```commandline
python predict.py  --case ... --pretrain_model_path ... --save_path ...
```
train: 
```commandline
python train.py --data_dir ... --save_dir ...
```

`checkpoints`，`checkpoints_1`：历史checkpoints存放路径，可忽略，运行train.py时重新定义


