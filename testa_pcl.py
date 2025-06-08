import pcl
import numpy as np


class Point:
    def __init__(self, x, y, z, r, g, b, classification, return_number, number_of_returns, z_duplicate):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.classification = classification
        self.return_number = return_number
        self.number_of_returns = number_of_returns
        self.z_duplicate = z_duplicate
        self.label = 0

    def __repr__(self):
        return (f"Point(x={self.x}, y={self.y}, z={self.z}, r={self.r}, "
                f"g={self.g}, b={self.b}, classification={self.classification}, "
                f"return_number={self.return_number}, number_of_returns={self.number_of_returns}, "
                f"z_duplicate={self.z_duplicate})")


def parse_points(file_path, count=0):
    points = []

    with open(file_path, 'r') as file:
        i = 0
        for line in file:
            i = i + 1
            # 去除行尾的换行符并分割字段
            if (count != 0):
                if i % count != 1:
                    continue

            parts = line.strip().split(',')

            if len(parts) >= 10:
                try:
                    # 将字符串转换为对应的数据类型
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    r = int(parts[3])
                    g = int(parts[4])
                    b = int(parts[5])
                    classification = float(parts[6])
                    return_number = float(parts[7])
                    number_of_returns = float(parts[8])
                    z_duplicate = float(parts[9])

                    # 创建Point对象并添加到列表中
                    points.append(
                        Point(x, y, z, r, g, b, classification, return_number, number_of_returns, z_duplicate))
                except ValueError as e:
                    print(f"解析错误: {e}，跳过此行: {line}")
            else:
                print(f"字段不足，跳过此行: {line}")

    return points


def convert_to_pcl_point_cloud(point_list):
    """
    将自定义 Point 对象列表转换为 PCL 点云结构
    :param point_list: list of Point
    :return: pcl.PointCloud
    """
    # 提取坐标数组 (Nx3)
    points_array = np.array([[p.x, p.y, p.z] for p in point_list], dtype=np.float32)

    # 创建 PCL 点云对象
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(points_array)

    return pcl_cloud


def extract_boundary_points_with_pcl(pcl_cloud, k=50, angle_threshold_degrees=45.0):
    """
    使用 PCL 的 BoundaryEstimation 提取边界点
    :param pcl_cloud: pcl.PointCloud 对象
    :param k: 法向量估计时使用的近邻数量
    :param angle_threshold_degrees: 判断是否为边界的夹角阈值（角度）
    :return: 边界点索引列表
    """
    # Step 1: 计算法向量
    normal_estimator = pcl_cloud.make_NormalEstimation()
    normal_estimator.set_KSearch(k)
    normals = normal_estimator.compute()

    # Step 2: 创建边界估计器
    boundary_estimator = pcl_cloud.make_BoundaryEstimation(
        normals,
        angle_threshold_radians=np.deg2rad(angle_threshold_degrees)
    )

    # Step 3: 执行边界点检测
    boundary_indices = boundary_estimator.get_boundary_points()

    return boundary_indices


def visualize_boundary_points_with_pcl(point_list, boundary_indices):
    """
    使用 Open3D 可视化边界点（PCL 提取结果）
    :param point_list: 原始 Point 对象列表
    :param boundary_indices: 边界点索引列表
    """
    import open3d as o3d

    points_array = np.array([[p.x, p.y, p.z] for p in point_list])
    colors = np.zeros_like(points_array)  # 默认灰色
    colors[boundary_indices] = [1, 0, 0]  # 边界点设为红色

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="PCL 边界点提取")


if __name__ == "__main__":
    file_path = "points.txt"
    point_list = parse_points(file_path, 50)
    pcl_cloud = convert_to_pcl_point_cloud(point_list)
    boundary_indices = extract_boundary_points_with_pcl(pcl_cloud, k=50, angle_threshold_degrees=45)
    print(f"共检测到 {len(boundary_indices)} 个边界点")

    visualize_boundary_points_with_pcl(point_list, boundary_indices)
