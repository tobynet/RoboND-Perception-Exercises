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
from sensor_msgs.msg import PointCloud2, PointField


def voxel_filter(cloud, leaf_size = 0.01):
    # Voxel Grid filter for input point cloud
    # Crate Voxel grid
    vox = cloud.make_voxel_grid_filter()

    # To obtain the downsampled point clouds 
    # 
    # voxel size (also known as `leaf`)
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)

    return vox.filter()

def pass_through_filter(cloud, filter_axis='z', axis_limit=(0.6, 1.1)):
    """
    Cut off with range using PassThrough filter
        filter_axis: axis
        axis_limit: range of cutting off
    Returns:
        filter: pcl.PCLPointCloud2
    """
    # Create a PassThrough filter object.
    passthrough = cloud.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_limit[0], axis_limit[1])

    return passthrough.filter()


def ransac_plane_segmentation(cloud, max_distance = 0.010):
    """
    Separate table and objects

    parameters:
        max_distance: threshold
    return:
        inliers, coefficients: pcl.PointIndices of table, pcl.ModelCoefficients
    """

    # RANSAC plane segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Setup parameters
    # 1..0.1 : xx
    # 0.01.. : OK!
    seg.set_distance_threshold(max_distance)

    # Obtain inliers and the model
    inliers, coefficients = seg.segment()
    return  inliers, coefficients


def euclidean_clustering(cloud, tolerance=0.001, cluster_range=(10,250)):
    """
    Euclidean Clustering using PCL with k-d tree

    returns:
        cluster_indices, white_cloud: pcl.PointIndices, XYZ Point Cloud
    """

    # Obtain XYZ only from XYZRGB nad convert it to kd-tree
    white_cloud = XYZRGB_to_XYZ(cloud)
    kdtree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    # Setup parameters:
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(cluster_range[0])
    ec.set_MaxClusterSize(cluster_range[1])
    
    # Search the k-d tree for clusters
    ec.set_SearchMethod(kdtree)

    # Extract indices for each of the discovered clusters
    indices = ec.Extract()
    return indices, white_cloud


def create_colored_cluster_cloud(cluster_indices, white_cloud):
    # Cluster by color and convert XYZ to XYZRGB
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
    # Assign axis and range to the passthrough filter object.
    cloud_filtered = pass_through_filter(cloud_filtered,
        filter_axis='z', axis_limit=(0.6, 1.1))
    # Cut off the near side of the table too.
    cloud_filtered = pass_through_filter(cloud_filtered,
        filter_axis='y', axis_limit=(-10.0, -1.39))

    # DONE: RANSAC Plane Segmentation
    inliers, _ = ransac_plane_segmentation(cloud_filtered)

    # DONE: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    rospy.loginfo(' cloud_objects: %d' % (cloud_objects.size))

    # DONE: Euclidean Clustering
    cluster_indices, white_cloud = euclidean_clustering(cloud_objects,
        tolerance=0.02, cluster_range=(100,15000))
    rospy.loginfo(' cluster_indices: %d, white_cloud: %d ' % (len(cluster_indices), white_cloud.size))


    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = create_colored_cluster_cloud(cluster_indices, white_cloud)
    rospy.loginfo(' colored cluster: %d' % (cluster_cloud.size))

    # DONE: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # DONE: Publish ROS messages
    rospy.loginfo(' Publish to objects')
    pcl_objects_pub.publish(ros_cloud_objects)

    rospy.loginfo(' Publish to table')
    pcl_table_pub.publish(ros_cloud_table)

    rospy.loginfo(' Publish to cluster_cloud')
    pcl_cluster_pub.publish(ros_cluster_cloud)

    rospy.loginfo('DONE Exercise-2')

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    # Object recognition
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
    detected_objects_pub.publish(detected_objects)


if __name__ == '__main__':
    rospy.loginfo('Initialize object recognition')

    # DONE: ROS node initialization
    rospy.init_node('recognition', anonymous=True)

    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)
    rospy.loginfo('Subscribed')


    # DONE: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for objects')
  
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for table')

    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    rospy.loginfo('Initialize Publisher for objects cluster')

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
