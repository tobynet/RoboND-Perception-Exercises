#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
from pcl_helper import *
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import pcl

# DONE: Define functions as required

def voxel_filter(cloud, leaf_size = 0.01):
    # Voxel Grid filter for input point cloud
    # ボクセルを生成
    vox = cloud.make_voxel_grid_filter()

    # ボクセルのグリッドサイズを設定
    # 大きいほど Point Cloud の点を減らせる
    # 1 -> increment..
    # voxel size (also known as `leaf`)
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)

    return vox.filter()

def pass_through_filter(cloud, filter_axis='z', axis_limit=(0.6, 1.1)):
    """
    欲しい範囲だけを切り取るフィルタ
        filter_axis: 切り取る軸
        axis_limit: テーブル下上のリミット
    Returns:
        filter
    """
    # Create a PassThrough filter object.
    # パススルー用フィルタをゲット
    passthrough = cloud.make_passthrough_filter()

    # ざっくり z軸でテーブルの上だけを切り取る
    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_limit[0], axis_limit[1])

    return passthrough.filter()


def ransac_plane_segmentation(cloud, max_distance = 0.010):
    """
    RANSAC で分離する
    parameters:
        max_distance: しきい値
    return:
        inliers, coefficients: 近い点群, 係数？
    """

    # RANSAC で机とオブジェクトを分離する
    # RANSAC plane segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # 判定するための、しきい値を設定する?
    # 1..0.1 : xx
    # 0.01.. : OK!
    seg.set_distance_threshold(max_distance)

    # inlier(推定したいものに近い側の点)とモデルを得る
    inliers, coefficients = seg.segment()
    return  inliers, coefficients


def euclidean_clustering(cloud, tolerance=0.001, cluster_range=(10,250)):
    """
    Euclidean Clustering using PCL with k-d tree

    returns:
        cluster_indices, white_cloud
    """
    # XYZ情報のみの オブジェクト用 Point Cloud を取得し、 PCL 用に kd-tree に変換。
    white_cloud = XYZRGB_to_XYZ(cloud)
    kdtree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    # オブジェクトの生成
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    # パラメータの設定:
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(cluster_range[0])
    ec.set_MaxClusterSize(cluster_range[1])
    
    # Search the k-d tree for clusters
    # k-d tree でクラスタリングする
    ec.set_SearchMethod(kdtree)

    # Extract indices for each of the discovered clusters
    # クラスタリングした点群の index を取り出す
    indices = ec.Extract()
    return indices, white_cloud


def create_colored_cluster_cloud(cluster_indices, white_cloud):
    # cluster 毎に色を分けつつ、 XYZ -> XYZRGB に変換。
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for _, indice in enumerate(indices):
            color_cluster_point_list.append([
                white_cloud[indice][0], white_cloud[indice][1], white_cloud[indice][2],
                rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    rospy.loginfo(' Begin pcl_callback...')

    # DONE: Convert ROS msg to PCL data(PointXYZRGB)
    cloud = ros_to_pcl(pcl_msg)

    # DONE: Voxel Grid Downsampling
    cloud_filtered = voxel_filter(cloud)

    # DONE: PassThrough Filter
    # ざっくり z軸でテーブルの上だけを切り取る
    cloud_filtered = pass_through_filter(cloud_filtered, 
        filter_axis='z', axis_limit=(0.6, 1.1))
    # テーブルの手前もオブジェクトに見えるので、切り取る
    cloud_filtered = pass_through_filter(cloud_filtered,
        filter_axis='y', axis_limit=(-10.0, -1.39))

    # DONE: RANSAC Plane Segmentation
    inliers, _ = ransac_plane_segmentation(cloud_filtered)

    # DONE: Extract inliers and outliers
    # フィルタでオブジェクト部分を取り除く(結果は ransanc の distance に依存する)
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    rospy.loginfo(' cloud_objects: %d' % (cloud_objects.size))

    # DONE: Euclidean Clustering
    cluster_indices, white_cloud = euclidean_clustering(cloud_objects,
        tolerance=0.02, cluster_range=(100,15000))
    rospy.loginfo(' cluster_indices: %d, white_cloud: %d ' % (len(cluster_indices), white_cloud.size))

    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # cluster 毎に色を分けつつ、 XYZ -> XYZRGB に変換。
    cluster_cloud = create_colored_cluster_cloud(cluster_indices, white_cloud)
    rospy.loginfo(' colored cluster: %d' % (cluster_cloud.size))

    # DONE: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # DONE: Publish ROS messages
    # topic へ送信
    rospy.loginfo(' Publish to objects')
    pcl_objects_pub.publish(ros_cloud_objects)

    rospy.loginfo(' Publish to table')
    pcl_table_pub.publish(ros_cloud_table)

    rospy.loginfo(' Publish to cluster_cloud')
    pcl_cluster_pub.publish(ros_cluster_cloud)

    rospy.loginfo(' DONE pcl_callback')




if __name__ == '__main__':

    rospy.loginfo('Initialize segmentation')
    # DONE: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # DONE: Create Subscribers
    # PointCloud 用のメッセージを受け取って処理する
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)
    rospy.loginfo('Subscribed')

    # DONE: Create Publishers
    # topic へメッセージを送信するためのもの
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for objects')
  
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for table')

    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for objects cluster')


    # Initialize color_list
    get_color_list.color_list = []

    # DONE: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
