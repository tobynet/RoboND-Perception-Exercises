#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import rospy
import pickle
import pcl

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import make_label
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import XYZRGB_to_XYZ, get_color_list, ros_to_pcl, pcl_to_ros, rgb_to_float

import sensor_msgs.point_cloud2 as pc2

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
        cluster_indices: インデックス
        white_cloud: Point Cloud XYZ only
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


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    rospy.loginfo('Begin pcl_callback...')

# Exercise-2 TODOs:

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
    # フィルタでオブジェクト群とテーブルを分離する(結果は ransanc の distance に依存する)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    rospy.loginfo(' cloud_objects: %d' % (cloud_objects.size))

    # DONE: Euclidean Clustering
    # オブジェクト毎に分割する
    cluster_indices, white_cloud = euclidean_clustering(cloud_objects,
        tolerance=0.02, cluster_range=(100,15000))
    rospy.loginfo(' cluster_indices: %d, white_cloud: %d ' % (len(cluster_indices), white_cloud.size))

    rospy.loginfo('DONE Exercise-2')

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    # 各オブジェクトを認識する
    for index, pts_list in enumerate(cluster_indices):
        #rospy.loginfo('  Detecting... : %d' % (index))
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        # DONE: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # DONE: complete this step just as is covered in capture_features.py
        # Compute the associated feature vector
        #rospy.loginfo('  type : %s' % (type(ros_cluster)))
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        # ラベルを publish する
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)


    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    # 
    # 認識したオブジェクトを publish する
    detected_objects_pub.publish(detected_objects)


if __name__ == '__main__':
    rospy.loginfo('Initialize object recognition')

    # DONE: ROS node initialization
    rospy.init_node('recognition', anonymous=True)

    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)
    rospy.loginfo('Subscribed')


    # DONE: Create Publishers
    # DONE: here you need to create two publishers
    # Call them object_markers_pub and detected_objects_pub
    # Have them publish to "/object_markers" and "/detected_objects" with 
    # Message Types "Marker" and "DetectedObjectsArray" , respectively
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    rospy.loginfo('Initialize Publisher for Markers')

    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)
    rospy.loginfo('Initialize Publisher for DetectedObjectsArray')


    # DONE: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # DONE: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
