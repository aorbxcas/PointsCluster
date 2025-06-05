# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
# import pcl


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
    for point in curved_surface_points:
        point.r, point.g, point.b = 0, 0, 255  # 蓝色 - 曲面结构

    # Step 4: 合并并可视化
    all_colored_points = planar_points + curved_surface_points
    visualize_points_open3d(all_colored_points)

    # 输出统计信息
    print(f"平面结构点数量: {len(planar_points)}")
    print(f"曲面结构点数量: {len(curved_surface_points)}")


# endregion

# region PCL边缘检测



def points_to_pcl(point_list):
    """
    将自定义 Point 对象列表转换为 pcl.PointCloud 实例
    :param point_list: list of `Point`
    :return: pcl.PointCloud
    """
    points_array = np.array([[p.x, p.y, p.z] for p in point_list], dtype=np.float32)
    cloud = pcl.PointCloud()
    cloud.from_array(points_array)
    return cloud

def detect_boundaries_with_pcl(cloud, radius=0.5, angle_threshold=0.7):
    """
    使用 PCL 进行边界检测
    :param cloud: pcl.PointCloud
    :param radius: 邻域搜索半径
    :param angle_threshold: 判断边界的阈值 (0~1)
    :return: 边界点索引列表
    """
    # 法向量估计
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(radius)  # 设置邻域搜索半径
    normals = ne.compute()

    # 边界检测
    boundary = cloud.make_BoundaryEstimation()
    boundary.set_InputNormals(normals)
    boundary.set_AngleThreshold(angle_threshold)
    boundaries = boundary.get_boundaries()  # 返回布尔数组：True 表示是边界点

    boundary_indices = [i for i, is_boundary in enumerate(boundaries) if is_boundary]
    return boundary_indices

def visualize_boundaries(original_points, boundary_indices):
    """
    将边界点染成红色并可视化
    :param original_points: 原始 Point 对象列表
    :param boundary_indices: 边界点索引列表
    """
    colored_points = [
        Point(p.x, p.y, p.z, p.r, p.g, p.b,
              p.classification, p.return_number, p.number_of_returns, p.z_duplicate)
        for p in original_points
    ]

    for idx in boundary_indices:
        if 0 <= idx < len(colored_points):
            colored_points[idx].r, colored_points[idx].g, colored_points[idx].b = 0, 255, 0  # 红色


def pcl_edge_detection_main(file_path):
    points = parse_points(file_path)  # 读取所有点
    cloud = points_to_pcl(points)     # 转换为 PCL 格式
    boundary_indices = detect_boundaries_with_pcl(cloud, radius=0.5, angle_threshold=0.7)

    print(f"检测到 {len(boundary_indices)} 个边界点")
    visualize_boundaries(points, boundary_indices)
    visualize_points_open3d(points)

# endregion

# 准确率、耗时等数据DebugLog、可行性


if __name__ == "__main__":
    file_path = "points.txt"  # 假设文本文件名为points.txt
    # 原图像
    points = parse_points(file_path, 0)

    pcl_edge_detection_main(file_path)

    # visualize_points_open3d(points)
    #
    #
    # # 区域生长算法主函数
    # points_region = parse_points(file_path, 100)
    # region_growing_main(points_region)
    #
    # # 随机采样拟合主函数
    # points_cluster = parse_points(file_path, 5)
    # ransac_planar_and_curved_clustering_main(points_cluster,0.5)

