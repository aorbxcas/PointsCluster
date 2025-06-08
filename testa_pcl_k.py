import pcl
import numpy as np
from pcl import PointCloud
from pcl import PointCloud_Visualizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def extract_boundaries(points):
    # 从点云数据中提取边界点
    # 首先将点云数据转换为numpy数组
    point_data = np.array([(point.x, point.y, point.z) for point in points], dtype=np.float32)

    # 创建PCL点云对象
    pcl_cloud = PointCloud()
    pcl_cloud.from_array(point_data)

    # 执行体素网格下采样
    fil = pcl_cloud.make_voxel_grid_filter()
    fil.set_leaf_size(0.1, 0.1, 0.1)
    pcl_cloud = fil.filter()

    # 计算法线估计
    ne = pcl_cloud.make_NormalEstimation()
    tree = pcl_cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RADIUS_SEARCH(0.1)
    normals = ne.compute()

    # 分割边界
    seg = pcl_cloud.make_segmenter_normals()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.05)
    seg.set_normal_distance_weight(0.05)
    seg.set_max_iterations(100)
    inliers = seg.segment()

    # 提取边界点
    boundaries = pcl_cloud.extract(inliers, negative=True)

    # 将边界点转换为numpy数组
    boundaries_data = boundaries.to_array()
    boundaries_points = [Point(x, y, z, 0, 0, 0, 0, 0, 0, 0) for x, y, z in boundaries_data]

    return boundaries_points


def visualize_boundaries(points, boundaries):
    # 将点云数据转换为numpy数组
    point_data = np.array([(point.x, point.y, point.z) for point in points], dtype=np.float32)
    boundaries_data = np.array([(point.x, point.y, point.z) for point in boundaries], dtype=np.float32)

    # 创建可视化对象
    vis = PointCloud_Visualizer()

    # 添加原始点云
    vis.addPointCloud(point_data, 'original')

    # 添加边界点云
    vis.addPointCloud(boundaries_data, 'boundaries')

    # 显示可视化
    vis.spin()

    # 保存图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_data[:, 0], point_data[:, 1], point_data[:, 2], c='b', s=1)
    ax.scatter(boundaries_data[:, 0], boundaries_data[:, 1], boundaries_data[:, 2], c='r', s=10)
    plt.savefig('boundaries.png', dpi=300)


if __name__ == "__main__":
    # 解析点云数据
    points = parse_points('points.csv')

    # 提取边界点
    boundaries = extract_boundaries(points)

    # 可视化边界点
    visualize_boundaries(points, boundaries)