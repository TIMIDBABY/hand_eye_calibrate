# 本项目基于 https://github.com/leo038/hand_eye_calibrate.git 制作

手眼标定的原理可参考：[机械臂手眼标定方法详解](https://blog.csdn.net/leo0308/article/details/141498200)；

需要先行安装Intel D435的驱动；

"in"表示眼在手上，"out"表示眼在手外；

data_collect_ur5能够自动连通210的ur5机械臂，并在拍照时自动获取机械臂的位姿；

collect_data_in和collect_data_out文件夹中都有示例图片，可以直接运行理解代码；

记得在各collect_data_*文件中修改存储路径：
image_save_path = "./collect_data_in/" or image_save_path = "./collect_data_out/"。

