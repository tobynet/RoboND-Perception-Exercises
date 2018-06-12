# -*- coding: utf-8 -*-
# Import PCL module
import pcl

# Load Point Cloud file
# シミュレーション用に Point Cloud のデータを読み込む 
cloud = pcl.load_XYZRGB('tabletop.pcd')

### Downsampling using Voxel Grid filter

# Voxel Grid filter for input point cloud
# Crate Voxel grid
vox = cloud.make_voxel_grid_filter()

# Setup the size of voxel grid
#LEAF_SIZE = 0.1  # too little points
LEAF_SIZE = 0.01  # voxel size (also known as `leaf`)
#LEAF_SIZE = 0.001 #(debbugin)  # too many points
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# To obtain the downsampled point clouds 
# 
# $ python RANSAC.py && pcl_viewer voxel_downsampled.pcd
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)


### Cut off with range using PassThrough filter

# Create a PassThrough filter object.
passthrough = cloud_filtered.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6    # Clip below the table 
axis_max = 1.1    # Clip the table top
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()

# Cut off the near side of the table too.
passthrough = cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('y')
passthrough.set_filter_limits(-10.0, -1.4)


# Finally use the filter function to obtain the resultant point cloud. 
# 
# $ python RANSAC.py && pcl_viewer pass_through_filtered.pcd
cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)


### Separate table and objects

# RANSAC plane segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Setup parameters
# 1..0.1 : xx
# 0.01.. : OK!
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# Obtain inliers and the model 
inliers, coefficients = seg.segment()


# Extract inliers
#
# $ python RANSAC.py && pcl_viewer extracted_inliers.pcd
extracted_inliers = cloud_filtered.extract(inliers, negative=False)

# Save pcd for table
# pcl.save(cloud, filename)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)


# Extract outliers 
# 
# $ python RANSAC.py && pcl_viewer extracted_outliers.pcd &
extracted_outliers = cloud_filtered.extract(inliers, negative=True)

# Save pcd for tabletop objects
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)
