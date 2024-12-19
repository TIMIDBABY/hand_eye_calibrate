# 本项目基于 https://github.com/leo038/hand_eye_calibrate.git 制作

# 适用于 Hnu 机器人学院灵巧操作实验室用于手眼标定的程序

需要先行安装D435的驱动；

"in"表示眼在手上，"out"表示眼在手外；

data_collect_ur5能够自动连通210的ur5机械臂，并在拍照时自动获取机械臂的位姿；

collect_data_in和collect_data_out都有示例图片，可以直接运行理解代码，并记得在各collect_data_*文件中修改存储路径：
image_save_path = "./collect_data_in/" or image_save_path = "./collect_data_out/"

# hand_eye_calibrate

该项目可以进行机器人的手眼标定。 


手眼标定的原理可参考：[机械臂手眼标定方法详解](https://blog.csdn.net/leo0308/article/details/141498200)


data_collect.py 为数据采集脚本， 可直接运行。 注意需要将机械臂位姿获取根据自己实际使用的机械臂进行修改。 


hand_eye_calibrate.py 为计算程序， 可根据采集的数据， 计算得到手眼转换关系矩阵。 


collect_data目录为示例数据， 可直接运行计算程序理解代码逻辑。 

