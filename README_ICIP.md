# 训练
python tools/train.py ./configs/icip/xxx.py

# 测试
* * config_path ckpt_path --format-only(格式化为提交格式)
需要首先在 dataset 的 data-test 中配置 test_outpath，同时在对应位置手动新建 output 目录
python tools/test.py configs/icip/xxx.py /root/code/mmsegmentation-dev/work_dirs/xxx/latest.pth --format-only

# 可视化
* * config_path ckpt_path --show-dir vis_path
需要在对应位置手动新建 vis 目录
python tools/test.py configs/icip/xxx.py /root/code/mmsegmentation-dev/work_dirs/xxx/latest.pth --show-dir /root/code/mmsegmentation-dev/work_dirs/xxx/vis