# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os
import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R

def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    return R, t

def save_matrix_to_txt(matrix, filename):
    """保存矩阵到txt文件，每行用中括号括起，参数之间用逗号分隔，整个矩阵用中括号括起"""
    # 确保目录存在
    save_dir = "./测量数据"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 完整的文件路径
    filepath = os.path.join(save_dir, f"{filename}.txt")
    
    # 保存矩阵，使用新的格式
    with open(filepath, "w", encoding="utf-8") as f:
        if matrix.ndim == 1:  # 对于一维数组
            matrix = matrix.reshape(1, -1)
        
        rows, cols = matrix.shape
        f.write("[")  # 整个矩阵的开始中括号
        f.write("\n")  # 换行，使格式更清晰
        
        for i in range(rows):
            f.write("[")  # 每行的开始中括号
            for j in range(cols):
                f.write(repr(matrix[i, j]))  # 使用 repr() 保留所有小数位
                if j < cols - 1:
                    f.write(", ")  # 数字间用逗号和空格分隔
            f.write("]")  # 每行的结束中括号
            if i < rows - 1:
                f.write(",\n")  # 行之间用逗号和换行分隔
            else:
                f.write("\n")  # 最后一行只需换行
        
        f.write("]")  # 整个矩阵的结束中括号
    
    print(f"已保存{filename}到：{filepath}")

def camera_calibrate(images_path):
    print("++++++++++ 开始相机标定 ++++++++++++++")

    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板角点的列数
    YY = 8   # 标定板角点的行数
    L = 0.035  # 标定板一格的长度 (单位: 米)

    # 设置亚像素角点寻找的参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 世界坐标系中的角点位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L  # 将单位长度赋值

    # 存储3D点和2D点
    obj_points = []  # 世界坐标系中的3D点
    img_points = []  # 图像坐标系中的2D点

    # 遍历文件夹内所有图片
    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')]

    if not image_files:
        print("没得图片了，退出程序...")
        return None, None, None, None

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for idx, image_file in enumerate(image_files):
        print(f"正在处理第 {idx + 1} 张图片: {image_file}")

        img = cv2.imread(image_file)
        if img is None:
            print(f"警告: 无法读取图片 {image_file}，跳过。")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        # 寻找棋盘角点
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

        if ret:
            obj_points.append(objp)

            # 亚像素角点优化
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)

            # 绘制角点并显示
            cv2.drawChessboardCorners(img, (XX, YY), corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)  # 停留0.5秒
        else:
            print("找不到角点！！！！！！！！！！！！！！！！！！！！")

    cv2.destroyAllWindows()

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    if ret:
        print("内参矩阵:\n", mtx)
        print("畸变系数:\n", dist)
        print("++++++++++ 相机标定完成 ++++++++++++++")
        return rvecs, tvecs, mtx, dist
    else:
        print("错误: 相机标定失败。")
        return None, None, None, None

def process_arm_pose(arm_pose_file):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm

def hand_eye_calibrate(images_path, arm_pose_file):
    """手眼标定实现"""
    # 获取相机标定结果
    rvecs, tvecs, camera_matrix, dist_coeffs = camera_calibrate(images_path=images_path)
    if rvecs is None:
        return None, None, None, None
    
    # 获取机械臂位姿
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file)
    
    # 进行手眼标定
    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    
    print("+++++++++++手眼标定完成+++++++++++++++")
    return R, t, camera_matrix, dist_coeffs

if __name__ == "__main__":
    images_path = "./collect_data_in"  #记得改
    arm_pose_file = "./collect_data_in/poses.txt"   #记得改

    R, t, mtx, dist = hand_eye_calibrate(images_path, arm_pose_file)
    
    if R is not None and t is not None:
        print("旋转矩阵：")
        print(R)
        print("平移向量：")
        print(t)
        
        # 保存所有矩阵到txt文件
        save_matrix_to_txt(R, "旋转矩阵")
        save_matrix_to_txt(t, "平移矢量")
        save_matrix_to_txt(mtx, "内参矩阵")
        save_matrix_to_txt(dist, "畸变系数")