#!/usr/bin/env python
import rospy
import torch
import numpy as np
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import pickle


class CustomData(PointCloud2):
    def __init__(self, point_cloud, indices, distance, fields, data, header):
        super(CustomData, self).__init__(data=data, fields=fields, header=header)
        # self.header = header
        self.point_cloud = point_cloud
        self.indices = indices
        self.distance = distance


if __name__ == "__main__":
    file = '/var/local/home/aburai/DATA/WADS2/sequences/11/velodyne/039498.bin'
    label_file = '/var/local/home/aburai/DATA/WADS2/sequences/11/labels/039498.label'
    kd_path = file.replace('velodyne', 'knn')[:-3] + 'pkl'
    data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

    pub = rospy.Publisher('spin', PointCloud2, queue_size=10)
    rospy.init_node('spin_publisher')
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'spin_lab'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)
              ]
    pc = point_cloud2.create_cloud(header, fields, data)
    annotated_data = np.fromfile(label_file,
                                 dtype=np.int32).reshape((-1, 1))
    with open(kd_path, 'rb') as f:
        try:
            ind = pickle.load(f)
            dist = pickle.load(f)
            # if ind.shape[1] > self.k:
            #     ind = ind[:, :self.k]
            #     dist = dist[:, :self.k]
            err = False
        except EOFError:
            err = True
    header1 = Header()
    header1.stamp = rospy.Time.now()
    c_data = CustomData(pc, ind, dist, fields, data, header1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(pc)
        rate.sleep()
