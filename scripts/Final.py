#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import os
import rospy
import actionlib
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


aug_dics = {}

image = np.array([])


# Subscribe to Camera Node and use CvBridge() for ROS-OPENCV compatibility
class camera_1:

	def __init__(self):
		self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)

	def callback(self,data):
		bridge = CvBridge()

		try:
			cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
			
		except CvBridgeError as e:
			rospy.logerr(e)
			
		image = cv_image
		pass_image(image)
		
		
# Loop through all the markers and augment each one
def pass_image(image):

    # while True:
	
    aruco_found = find_aruco_markers(image)

    if len(aruco_found[0]) != 0:
        for bbox, id in zip(aruco_found[0], aruco_found[1]):
            if int(id) in aug_dics.keys():
                image = augment_aruco(bbox, id, image, aug_dics[int(id)])
        
        
    cv2.imshow("Image", image)
    cv2.waitKey(1)           
        

def load_aug_images(path):
    my_list = os.listdir(path)
    no_of_markers = len(my_list)

    print("Total Number of Markers Detected: ", no_of_markers)

    for imgPath in my_list:
        key = int(os.path.splitext(imgPath)[0])
        img_aug = cv2.imread(f'{path}/{imgPath}')
        aug_dics[key] = img_aug


def find_aruco_markers(image, marker_size=6, total_markers=250, draw=True):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_param)

    if draw:
        aruco.drawDetectedMarkers(image, bboxs)

    return [bboxs, ids]


def augment_aruco(bbox, id, image, img_aug, draw_id=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = img_aug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    img_out = cv2.warpPerspective(img_aug, matrix, (image.shape[1], image.shape[0]))
    cv2.fillConvexPoly(image, pts1.astype(int), (0, 0, 0))
    img_out = image + img_out

    if draw_id:
        cv2.putText(img_out, str(id), (int(bbox[0][0][0]), int(bbox[0][0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return img_out


def main():
	
	load_aug_images("Markers")
	
	camera_1()
        
	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Shutting down")
  
	cv2.destroyAllWindows() 
	
	

if __name__ == '__main__':
	rospy.init_node('camera_read', anonymous=False)
	main()
