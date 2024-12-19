# coding=utf-8
"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂基座坐标系的 旋转矩阵和平移向量
A*X=Z*B
其中：
A: 机械臂末端相对于基座的变换矩阵
X: 标定板相对于相机的变换矩阵
Z: 相机相对于机械臂基座的变换矩阵
B: 标定板相对于机械臂末端的变换矩阵
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
    """将位姿转换为齐次变换矩阵"""
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    return R, t

def camera_calibrate(images_path):
    """相机标定函数，保持不变"""
    print("++++++++++ 开始相机标定 ++++++++++++++")

    XX = 11  # 标定板角点的列数
    YY = 8   # 标定板角点的行数
    L = 0.035  # 标定板一格的长度 (单位: 米)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L

    obj_points = []  # 世界坐标系中的3D点
    img_points = []  # 图像坐标系中的2D点

    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')]

    if not image_files:
        print("没有找到图片，退出程序...")
        return None, None

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for idx, image_file in enumerate(image_files):
        print(f"正在处理第 {idx + 1} 张图片: {image_file}")

        img = cv2.imread(image_file)
        if img is None:
            print(f"警告: 无法读取图片 {image_file}，跳过。")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)

            cv2.drawChessboardCorners(img, (XX, YY), corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)
        else:
            print("未能找到角点！")

    cv2.destroyAllWindows()

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
    """处理机械臂的pose文件"""
    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm

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

def hand_eye_calibrate(images_path, arm_pose_file):
    """眼在手外标定实现"""
    # 获取相机标定结果
    rvecs, tvecs, camera_matrix, dist_coeffs = camera_calibrate(images_path=images_path)
    if rvecs is None:
        return None, None, None, None
    
    # 保存内参矩阵和畸变系数
    save_matrix_to_txt(camera_matrix, "内参矩阵")
    save_matrix_to_txt(dist_coeffs, "畸变系数")
    
    # 获取机械臂位姿
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file)
    
    # 使用CALIB_HAND_EYE_ROBOT_WORLD方法进行标定
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_arm, t_arm,
        [cv2.Rodrigues(rvec)[0] for rvec in rvecs],
        tvecs,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    
    print("+++++++++++眼在手外标定完成+++++++++++++++")
    return R_cam2base, t_cam2base, camera_matrix, dist_coeffs

if __name__ == "__main__":
    images_path = "./collect_data"  
    arm_pose_file = "./collect_data/poses.txt"  

    R, t, mtx, dist = hand_eye_calibrate(images_path, arm_pose_file)
    
    if R is not None and t is not None:
        print("相机相对于机械臂基座的旋转矩阵：")
        print(R)
        print("相机相对于机械臂基座的平移向量：")
        print(t)
        
        # 保存旋转矩阵和平移向量
        save_matrix_to_txt(R, "旋转矩阵")
        save_matrix_to_txt(t, "平移矢量")