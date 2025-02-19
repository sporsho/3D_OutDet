#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class CustomData(PointCloud2):
    def __init__(self, point_cloud, indices, distance, fields, data, header):
        super(CustomData, self).__init__(data=data, fields=fields, header=header)
        # self.header = header
        self.point_cloud = point_cloud
        self.indices = indices
        self.distance = distance


if __name__ == "__main__":
    file = '/var/local/home/rais/3D_OutDet/sample_data/sample/sample.bin'
    label_file = '/var/local/home/rais/3D_OutDet/sample_data/sample/sample.label'
    data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

    pub = rospy.Publisher('OutDet', PointCloud2, queue_size=10)
    rospy.init_node('OutDet_publisher')
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'out_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)
              ]
    pc = point_cloud2.create_cloud(header, fields, data)
    annotated_data = np.fromfile(label_file,
                                 dtype=np.int32).reshape((-1, 1))

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        pub.publish(pc)
        rate.sleep()
