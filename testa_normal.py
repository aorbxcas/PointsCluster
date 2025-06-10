import time

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


# region 点云数据读取

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

def visualize_points_open3d(points):
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 提取坐标和颜色
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    colors_array = np.array([[p.r / 255.0, p.g / 255.0, p.b / 255.0] for p in points])  # RGB归一化到[0,1]

    # 设置点云数据
    point_cloud.points = o3d.utility.Vector3dVector(points_array)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_array)

    # 打开交互式窗口查看点云

    o3d.visualization.draw_geometries([point_cloud], window_name="点云可视化 - Open3D")


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

def points_to_open3d(point_list):
    """
    将自定义 Point 对象列表转换为 open3d.geometry.PointCloud 实例
    :param point_list: list of `Point`
    :return: open3d.geometry.PointCloud
    """
    points_array = np.array([[p.x, p.y, p.z] for p in point_list])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_array)
    return cloud
# endregion

# region 区域生长算法
# 计算两点之间的欧氏距离
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# 计算RGB距离黑色的距离
def rgb_distance_to_black(point):
    # 黑色为 (0, 0, 0)，计算当前点与黑色的距离
    return np.sqrt(point.r ** 2 + point.g ** 2 + point.b ** 2)


def filter_black_points(points, threshold=10.0):
    """
    过滤掉颜色接近黑色的点
    :param points: Point对象列表
    :param threshold: RGB距离黑色的阈值
    :return: 过滤后的非黑色点列表
    """
    filtered_points = []
    for point in points:
        # 计算当前点与黑色的距离
        distance_to_black = rgb_distance_to_black(point)
        if distance_to_black >= threshold:
            filtered_points.append(point)
    return filtered_points


# 计算点的曲率
def compute_curvature(points, radius=1.0):
    """
    计算点云中每个点的曲率
    :param points: Point对象列表
    :param radius: 邻域搜索半径
    :return: 曲率数组 (每个点的曲率值)
    """
    # 构建KDTree加速邻域搜索
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    kdtree = KDTree(points_array)

    curvatures = []

    for i, point in enumerate(points):
        # 查询邻域点索引
        indices = kdtree.query_radius([points_array[i]], r=radius)[0]

        if len(indices) < 3:
            # 点太少无法估计曲率
            curvatures.append(0.0)
            continue

        # 获取邻域点坐标
        neighbors = points_array[indices]
        centroid = np.mean(neighbors)  # 邻域中心点
        diff = neighbors - centroid  # 偏移向量

        # 协方差矩阵
        cov_matrix = np.cov(diff)

        # 特征值分解
        eigenvalues = np.linalg.eigvalsh(cov_matrix)  # 只需要特征值
        eigenvalues.sort()  # 小 -> 大

        # 最小特征值对应表面变化最小的方向，曲率定义为 λ0 / (λ0 + λ1 + λ2)
        curvature = eigenvalues[0] / eigenvalues.sum()
        curvatures.append(curvature)

        point.classification = curvature

    return np.array(curvatures)


# 判断是否满足区域生长条件
def is_similar(p1, p2, curvature_threshold, rgb_threshold, position_threshold):
    curvature_diff = abs(p1.classification - p2.classification)  # 假设classification代表曲率
    rgb_diff = abs(rgb_distance_to_black(p1) - rgb_distance_to_black(p2))
    position_diff = euclidean_distance(p1, p2)

    return curvature_diff < curvature_threshold and rgb_diff < rgb_threshold and position_diff < position_threshold


def get_cluster_curvature_stats(points, labels):
    """
    计算每个聚类的平均曲率
    :param points: Point对象列表
    :param labels: 每个点所属的聚类ID
    :return: 字典 {cluster_id: avg_curvature}
    """
    cluster_curvatures = {}
    cluster_points_count = {}

    for point, label in zip(points, labels):
        if label not in cluster_curvatures:
            cluster_curvatures[label] = 0.0
            cluster_points_count[label] = 0
        cluster_curvatures[label] += point.classification
        cluster_points_count[label] += 1

    # 计算平均曲率
    for label in cluster_curvatures:
        cluster_curvatures[label] /= cluster_points_count[label]

    return cluster_curvatures


def classify_clusters_by_curvature(points, labels, curvature_threshold=0.1):
    """
    根据平均曲率将聚类分为两类：
    - 0: 平面类（如长方体）
    - 1: 曲面类
    """
    cluster_avg_curvature = get_cluster_curvature_stats(points, labels)

    # 对每个点重新分配类别标签
    final_labels = []
    for point, label in zip(points, labels):
        avg_curv = cluster_avg_curvature.get(label, 0.0)
        if avg_curv < curvature_threshold:
            final_labels.append(0)  # 平面类
        else:
            final_labels.append(1)  # 曲面类

    return final_labels


def region_growing_clustering(points, curvature_threshold=0.5, rgb_threshold=10.0, position_threshold=1.0,
                              min_points=10):
    num_points = len(points)
    labels = [-1] * num_points  # -1 表示未被访问
    cluster_id = 0

    # 构建KDTree以加速邻域搜索
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    kdtree = KDTree(points_array)

    for i in range(num_points):
        if labels[i] != -1:
            continue

        # 初始化一个新的聚类
        queue = [i]
        labels[i] = cluster_id
        current_cluster_size = 1

        while queue:
            current_idx = queue.pop(0)

            # 使用KDTree找到邻近点
            indices = kdtree.query_radius([points_array[current_idx]], r=position_threshold)[0]

            for neighbor_idx in indices:
                if labels[neighbor_idx] == -1 and is_similar(points[current_idx], points[neighbor_idx],
                                                             curvature_threshold, rgb_threshold, position_threshold):
                    labels[neighbor_idx] = cluster_id
                    queue.append(neighbor_idx)
                    current_cluster_size += 1

        # 如果聚类大小大于最小点数，则增加cluster_id
        if current_cluster_size >= min_points:
            cluster_id += 1

    return labels


def update_points_colors(points, final_labels):
    """
    根据分类结果更新每个点的RGB颜色
    :param points: Point对象列表
    :param final_labels: 每个点对应的类别标签 (0 或 1)
    :return: None (原地修改points)
    """
    for point, label in zip(points, final_labels):
        if label == 0:  # 平面类 -> 红色
            point.r, point.g, point.b = 255, 0, 0
        elif label == 1:  # 曲面类 -> 蓝色
            point.r, point.g, point.b = 0, 0, 255
        else:  # 可选：未知类别 -> 灰色
            point.r, point.g, point.b = 128, 128, 128


def region_growing_main(points):
    # 摘除接近黑色的点
    # points = filter_black_points(points,200)
    # 计算曲率
    compute_curvature(points)
    # 区域生长聚类法
    labels = region_growing_clustering(points, curvature_threshold=0.5, rgb_threshold=10.0, position_threshold=1.0)
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"检测到 {len(unique_labels)} 个聚类:")
    final_labels = classify_clusters_by_curvature(points, labels, curvature_threshold=0.2)
    # 输出统计信息
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    print(f"最终分类结果（0: 平面类 / 1: 曲面类）:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个点")
    update_points_colors(points, final_labels)
    visualize_points_open3d(points)


# endregion

# region 随机采样拟合
def ransac_fit_plane(points, distance_threshold=0.05):
    """
    使用 RANSAC 拟合一个最优平面，并返回属于该平面的点索引
    :param points: Point 对象列表
    :param distance_threshold: 判断点是否属于平面的距离阈值
    :return: 平面参数 和 属于该平面的点索引列表
    """
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)

    # 使用 RANSAC 拟合平面
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=3,
                                                     num_iterations=100)
    return plane_model, inliers


def ransac_planar_and_curved_clustering_main(points, distance_threshold=0.05):
    """
    使用 RANSAC 将点云分为两类：
    - 类别 0：平面结构（如甲板）
    - 类别 1：曲面立体结构
    """
    # Step 1: 拟合平面模型
    _, inliers = ransac_fit_plane(points, distance_threshold=distance_threshold)

    # Step 2: 提取平面点和非平面点
    planar_points = [points[i] for i in inliers]
    curved_surface_points = [points[i] for i in range(len(points)) if i not in inliers]

    # Step 3: 设置颜色
    for point in planar_points:
        point.r, point.g, point.b = 255, 0, 0  # 红色 - 平面结构
        point.label = 1
    for point in curved_surface_points:
        point.r, point.g, point.b = 0, 0, 255  # 蓝色 - 曲面结构
        point.label = 2

    # Step 4: 合并并可视化
    all_colored_points = planar_points + curved_surface_points
    visualize_points_open3d(all_colored_points)

    # 输出统计信息
    print(f"平面结构点数量: {len(planar_points)}")
    print(f"曲面结构点数量: {len(curved_surface_points)}")


# endregion

# region 法向量边界检测
def draw_point_cloud_with_normals_colored(cloud):
    """
    可视化点云，并根据法向量方向为每个点着色
    :param cloud: open3d.geometry.PointCloud 对象，必须已经计算了法向量
    """
    normals = np.asarray(cloud.normals)

    # 法向量归一化（确保长度为1）
    normals_normalized = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    # 将法向量映射到 [0, 1] 范围作为 RGB 颜色
    colors = (normals_normalized + 1) / 2  # [-1,1] -> [0,1]

    # 创建新点云并设置颜色
    colored_cloud = cloud.select_by_index(range(len(cloud.points)))
    colored_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 可视化彩色点云
    o3d.visualization.draw_geometries([colored_cloud], window_name="点云 - 法向量颜色映射")


def draw_point_cloud_with_normals(cloud, scale=0.1):
    """
    可视化点云并绘制每个点的法向量（箭头）
    :param cloud: open3d.geometry.PointCloud 对象，必须已经计算了法向量
    :param scale: 箭头长度缩放因子
    """
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    # 创建一个 LineSet 来表示所有法向量
    lines = []
    colors = []
    for i in range(len(points)):
        p = points[i]
        n = normals[i]
        lines.append([p, p + scale * n])  # 法向量方向的线段
        colors.append([1, 0, 0])  # 红色表示法向量

    # 将线段转换为 LineSet
    line_sets = o3d.geometry.LineSet()
    all_points = np.vstack([line for line in lines])
    line_indices = np.array([[i*2, i*2+1] for i in range(len(lines))])
    line_sets.points = o3d.utility.Vector3dVector(all_points)
    line_sets.lines = o3d.utility.Vector2iVector(line_indices)
    line_sets.colors = o3d.utility.Vector3dVector(colors)

    # 显示原始点云 + 法向量箭头
    o3d.visualization.draw_geometries([cloud, line_sets], window_name="点云与法向量")


def detect_boundaries_with_open3d(cloud, radius, angle_threshold):
    # 计算法向量
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    cloud.orient_normals_to_align_with_direction(
        orientation_reference=np.array([0, 0, 1])  # 注意参数名是 orientation_reference
    )
    # 可视化法向量颜色映射
    draw_point_cloud_with_normals_colored(cloud)

    # 构建 KDTree 查询结构
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    normals = np.asarray(cloud.normals)
    boundary_indices = []

    for i in range(len(cloud.points)):
        _, idxs, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
        if len(idxs) < 3:
            continue

        current_normal = normals[i]
        angles = [abs(np.dot(current_normal, normals[j])) for j in idxs if j != i]

        avg_angle = np.mean(angles)
        # 判断是否存在明显差异的邻域点
        avg_angle = np.mean(angles)
        if avg_angle < angle_threshold:
            boundary_indices.append(i)

    print(f"共检测到 {len(boundary_indices)} 个边界点")
    return boundary_indices

def visualize_boundary_points(cloud, boundary_indices):
    """
    将边界点高亮显示（红色），其余点为灰色
    """
    points = np.asarray(cloud.points)
    colors = np.zeros_like(points)  # 默认灰色
    colors[:] = [0.5, 0.5, 0.5]

    # 边界点设为红色
    colors[boundary_indices] = [1, 0, 0]

    boundary_pcd = o3d.geometry.PointCloud()
    boundary_pcd.points = o3d.utility.Vector3dVector(points)
    boundary_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([boundary_pcd], window_name="点云 - 边界点高亮显示")


#endregion

# region Alhpa Shape边界检测
from scipy.spatial import Delaunay
from collections import defaultdict


def alpha_shape(points, alpha):
    """
    使用 Alpha Shape 算法提取三维点云边界（基于 Delaunay 三角剖分）
    :param points: Point 对象列表
    :param alpha: 控制边界紧密程度的参数
    :return: 边界点索引列表
    """
    # 提取坐标数组
    points_array = np.array([[p.x, p.y, p.z] for p in points])

    # 构建 Delaunay 三角剖分
    tetra = Delaunay(points_array)

    # 存储每条边对应的两个四面体
    edge_map = defaultdict(list)

    # 遍历所有四面体的三角面
    boundary_faces = []
    for t in tetra.simplices:
        # 四个顶点
        vertices = points_array[t]

        # 计算该四面体的外接球半径
        A, B, C, D = vertices
        AB = B - A
        AC = C - A
        AD = D - A
        normal = np.cross(AB, AC)
        denom = 2 * np.linalg.norm(normal)
        if denom < 1e-8:
            continue
        center = A + (np.linalg.det(np.array([AD, AC, normal])) / denom,
                      np.linalg.det(np.array([AB, AD, normal])) / denom,
                      np.linalg.det(np.array([AB, AC, AD])) / denom)
        radius = np.linalg.norm(center - A)

        # 判断是否属于边界四面体（即至少一个邻域缺失）
        face_indices = [
            (t[0], t[1], t[2]),
            (t[0], t[1], t[3]),
            (t[0], t[2], t[3]),
            (t[1], t[2], t[3]),
        ]
        for face in face_indices:
            sorted_face = tuple(sorted(face))
            edge_map[sorted_face].append(t)

    # 收集所有只出现一次的面（即边界上的面）
    boundary_faces = []
    for face, tetras in edge_map.items():
        if len(tetras) == 1:
            # 检查该三角面的外接圆半径是否 <= alpha
            A, B, C = points_array[list(face)]
            face_center = (A + B + C) / 3
            r = max(np.linalg.norm(A - face_center),
                    np.linalg.norm(B - face_center),
                    np.linalg.norm(C - face_center))
            if r <= alpha:
                boundary_faces.append(face)
            else:
                print("出现不符合alhpa半径的边界点，外接圆半径:" + r + "\n")

    # 提取所有出现在边界面上的点索引
    boundary_point_indices = set()
    for face in boundary_faces:
        for idx in face:
            boundary_point_indices.add(idx)
    print("获取符合alhpa半径的边界点数量:" + str(len(boundary_point_indices)) + "\n")

    return list(boundary_point_indices)


def visualize_alpha_shape_boundary(points, boundary_faces):
    """
    可视化 Alpha Shape 提取的边界
    :param points: Point 对象列表
    :param boundary_faces: 由 alpha_shape 返回的边界三角面列表
    """
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    colors = np.zeros_like(points_array)
    colors[:, 1] = 255  # 设置为绿色

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # 创建三角网格表示边界
    triangles = np.array(boundary_faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_array)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 设置颜色
    mesh.paint_uniform_color([0, 1, 0])  # 绿色边界

    # 可视化
    o3d.visualization.draw_geometries([mesh, point_cloud], window_name="Alpha Shape 边界提取")


def visualize_alpha_shape_boundary_only_points(points, boundary_point_indices):
    """
    可视化 Alpha Shape 提取的边界点，不显示三角面
    :param points: Point 对象列表
    :param boundary_point_indices: 边界点索引列表
    """
    points_array = np.array([[p.x, p.y, p.z] for p in points])
    colors = np.array([[0.0, 0.0, 0.0]] * len(points_array))  # 默认灰色
    colors[boundary_point_indices] = [1, 0, 0]  # 边界点设为红色

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud], window_name="Alpha Shape 边界点提取（仅点）")


# endregion

# region 经纬线扫描法（2D）

def load_point_cloud(file_path):
    point_cloud = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 10:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    r = int(parts[3])
                    g = int(parts[4])
                    b = int(parts[5])
                    classification = int(parts[6])
                    return_number = int(parts[7])
                    number_of_returns = int(parts[8])
                    z_duplicate = float(parts[9])
                    point = Point(x, y, z, r, g, b, classification, return_number, number_of_returns, z_duplicate)
                    point_cloud.append(point)
                except ValueError:
                    print(f"Error parsing line: {line}")
    return point_cloud

def find_point_cloud_boundary(point_cloud, interval=0.1):
    x_coords = [point.x for point in point_cloud]
    y_coords = [point.y for point in point_cloud]
    z_coords = [point.z for point in point_cloud]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    x_intervals = np.arange(min_x, max_x + interval, interval)
    y_intervals = np.arange(min_y, max_y + interval, interval)
    z_intervals = np.arange(min_z, max_z + interval, interval)

    boundary_points = []

    for x in x_intervals:
        points_at_x = [point for point in point_cloud if abs(point.x - x) < interval / 2]
        if points_at_x:
            min_y_point = min(points_at_x, key=lambda p: p.y)
            max_y_point = max(points_at_x, key=lambda p: p.y)
            boundary_points.append((min_y_point.x, min_y_point.y))
            boundary_points.append((max_y_point.x, max_y_point.y))

    for y in y_intervals:
        points_at_y = [point for point in point_cloud if abs(point.y - y) < interval / 2]
        if points_at_y:
            min_x_point = min(points_at_y, key=lambda p: p.x)
            max_x_point = max(points_at_y, key=lambda p: p.x)
            boundary_points.append((min_x_point.x, min_x_point.y))
            boundary_points.append((max_x_point.x, max_x_point.y))

    boundary_points = list(set(boundary_points))  # Remove duplicates

    return boundary_points

def visualize_point_cloud_and_boundary(point_cloud, boundary_points):
    x_coords = [point.x for point in point_cloud]
    y_coords = [point.y for point in point_cloud]
    boundary_x = [point[0] for point in boundary_points]
    boundary_y = [point[1] for point in boundary_points]

    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c='blue', s=1, label='Point Cloud')
    plt.scatter(boundary_x, boundary_y, c='red', s=20, label='Boundary Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud and Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# endregion

if __name__ == "__main__":

    start_time = time.time()

    file_path = "points.txt"  # 假设文本文件名为points.txt
    # 原图像
    count = 5
    points = parse_points(file_path, count)
    print("点云预处理中，点云稠密度：", 1/count)
    time_pre = time.time() - start_time
    print("点云预处理完成，耗时：", time_pre)

    # region 区域生长算法点云分割
    # points_region = parse_points(file_path, 100)
    # _region_growing_main(points_region)
    # endregion

    # region 随机采样法点云分割
    ransac_planar_and_curved_clustering_main(points, 0.5)
    points_cluster_1 = [point for point in points if point.label == 1]
    points_cluster_2 = [point for point in points if point.label == 2]
    cloud = points_to_open3d(points_cluster_1)  # 转换为 Open3D 格式
    #endregion

    time_cut = time.time() - time_pre - start_time
    print("点云分割完成，耗时：", time_cut)

    # region 法向量边界检测
    # # 检测边界点
    # boundary_indices = detect_boundaries_with_open3d(cloud, radius=2.0, angle_threshold=0.8)
    # # 高亮显示边界点
    # visualize_boundary_points(cloud, boundary_indices)
    # endregion

    # region alphaShape边界检测
    # boundary_faces = alpha_shape(points_cluster_1, alpha=100.0)
    # print(f"检测到 {len(boundary_faces)} 个边界三角面")
    # visualize_alpha_shape_boundary_only_points(points_cluster_1, boundary_faces)
    # endregion

    # region 经纬线、投影边界检测
    boundary_points = find_point_cloud_boundary(points_cluster_1, interval=0.1)
    visualize_point_cloud_and_boundary(points_cluster_1, boundary_points)
    # endregion

    time_boundary = time.time() - time_cut - start_time
    print("边界检测完成，耗时：", time_boundary)

    print("总耗时：", time.time() - start_time)

