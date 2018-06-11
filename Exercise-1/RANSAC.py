# -*- coding: utf-8 -*-
# Import PCL module
import pcl

# Load Point Cloud file
# シミュレーション用に Point Cloud のデータを読み込む 
cloud = pcl.load_XYZRGB('tabletop.pcd')

### Voxel Grid filter  で、ボクセル状にサンプル数を減らす

# Voxel Grid filter for input point cloud
# ボクセルを生成
vox = cloud.make_voxel_grid_filter()

# ボクセルのグリッドサイズを設定
# 大きいほど Point Cloud の点を減らせる
# 1 -> increment..
LEAF_SIZE = 0.1  # voxel size (also known as `leaf`)
#LEAF_SIZE = 0.001 #(debbugin)  # voxel size (also known as `leaf`)
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# To obtain the downsampled point clouds 
# ボクセル状にフィルタをかける。
# LEAF_SIZE が 大きいほど Point Cloud の点を減らせる
# 
# $ python RANSAC.py && pcl_viewer voxel_downsampled.pcd
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)


### PassThrough filter 欲しい範囲だけを切り取るフィルタ

# Create a PassThrough filter object.
# パススルー用フィルタをゲット
passthrough = cloud_filtered.make_passthrough_filter()

# ざっくり z軸でテーブルの上だけを切り取る
# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6    # テーブル下のリミット
axis_max = 1.1    # テーブル上のリミット
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()

# テーブルの手前もオブジェクトに見えるので、切り取る
passthrough = cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('y')
passthrough.set_filter_limits(-10.0, -1.4)


# Finally use the filter function to obtain the resultant point cloud. 
# フィルタの適用
# 
# $ python RANSAC.py && pcl_viewer pass_through_filtered.pcd
cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)


# RANSAC で机とオブジェクトを分離する
# RANSAC plane segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# 判定するための、しきい値を設定する?
# 1..0.1 : xx
# 0.01.. : OK!
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# inlier(推定したいものに近い側の点)とモデルを得る
inliers, coefficients = seg.segment()


# Extract inliers
# フィルタでオブジェクト部分を取り除く(結果は、上記の distance に依存する)
#
# $ python RANSAC.py && pcl_viewer extracted_inliers.pcd
extracted_inliers = cloud_filtered.extract(inliers, negative=False)

# Save pcd for table
# pcl.save(cloud, filename)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)


# Extract outliers
# フィルタで机を取り除く(結果は、上記の distance に依存する)
# 
# $ python RANSAC.py && pcl_viewer extracted_outliers.pcd &
extracted_outliers = cloud_filtered.extract(inliers, negative=True)

# Save pcd for tabletop objects
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)\


