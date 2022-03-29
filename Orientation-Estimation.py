import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def get_object_feat():
  ret, frame = vid.read()
  # frame = cv2.cvtColor(frame) # we don't need to take grey scale and 
  # blur it as it degrades performance for our application
  roi = cv2.selectROI(windowName="ROI", img=frame, showCrosshair=True, 
  fromCenter=False)
  x, y, w, h = roi
  init_frame = frame[y:y + h, x:x + w]
  # init_frame = cv2.GaussianBlur(init_frame,(9,9),0)
  cv2.imshow("init_frame", init_frame)
  cv2.waitKey(3000)
  kp, des = orb.detectAndCompute(init_frame, None)
  cv2.destroyAllWindows()
  return kp, des, frame, init_frame, x, y

def matcher(kp1, des1, kp2, des2, x, y):
  matches = bf.knnMatch(des1, des2, k=2)
  pts1 = []
  pts2 = []
  good = []
  min_w = 1200
  min_h = 1200
  max_w = 0
  max_h = 0
  for m, n in matches:
    if m.distance < 0.95*n.distance:
      good.append([m])
    if (min_w < kp2[m.trainIdx].pt[0]):
      min_w = kp2[m.trainIdx].pt[0]
    elif (max_w > kp2[m.trainIdx].pt[0]):
      max_w = kp2[m.trainIdx].pt[0]
    if (min_h < kp2[m.trainIdx].pt[1]):
      min_h = kp2[m.trainIdx].pt[1]
    elif (max_h > kp2[m.trainIdx].pt[1]):
      max_h = kp2[m.trainIdx].pt[1]
# print(m.distance)print(temp_co)
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append([kp1[m.queryIdx].pt[0]+x, p1[m.queryIdx].pt[1]+y])
  frame = [int(min_w), int(min_h), int(max_w), int(max_h)]
  pts1 = np.asarray(pts1)
  pts2 = np.asarray(pts2)
  return pts1, pts2, frame, good

def get_pose(pts1, pts2, cam_mat):
  E, mask = cv2.findEssentialMat(pts1, pts2, cam_mat)
  R = cv2.recoverPose(E, pts1, pts2, cam_mat)
  return R

if __name__ == "__main__":
  # Object Definations
  url = 'https://192.168.93.187:8080/video'
  vid = cv2.VideoCapture(url)
  orb = cv2.ORB_create(nfeatures=10000000)
  bf = cv2.BFMatcher()
  # Camera params
  cam_mat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  kp1, des1, last_frame,update_template, x, y = get_object_feat()

  # initial rotation matrix will be identity
  R_old = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  while (1):
    if cv2.waitKey(1) & 0xFF == ord('q'):
      vid.release()
      cv2.destroyAllWindows()
      break
    
    ret, frame = vid.read()

    # cv2.imshow("raw", frame)
    # cv2.waitKey(1)
    kp2, des2 = orb.detectAndCompute(frame, None)
    pts1, pts2, temp_co, good = matcher(kp1, des1, kp2, des2, x, y)
    # img3 = cv2.drawMatchesKnn(update_template, kp1, frame, kp2, good, None, flags=2)
    # cv2.waitKey(1)
    copy = frame
    cv2.rectangle(copy, (temp_co[0], temp_co[1]), (temp_co[2], 
    temp_co[3]), (0, 0, 225), 2)
    # cv2.imshow("good", img3)
    #cv2.imshow("copy", copy)
    cv2.waitKey(1)
    print(temp_co)
    pose = get_pose(pts1, pts2, cam_mat)
    R_new = np.dot(R_old, pose[1])
    #print(R_new)
    # updating bounding box
    y_add = int((temp_co[3]+temp_co[1])/8)
    x_add = int((temp_co[2]+temp_co[0])/8)
    update_template = frame[temp_co[3]-y_add:temp_co[1]+y_add, 
    temp_co[2]-x_add:temp_co[0]+x_add]
    # cv2.imshow('up', update_template)
    # cv2.waitKey(1)
    # print("update_template")
    # cv2.waitKey(1)

    kp1, des1 = orb.detectAndCompute(frame, None)
    x = 0#temp_co[2]-x_add
    y = 0#temp_co[3]-y_add
    # print(x, y)

    # get angles from rotational matrix
    R_temp = R.from_matrix(R_new)
    Euler_angles = R_temp.as_euler('zyx', degrees=True)
    print("angles")
    print(Euler_angles[2])
    cv2.putText(copy, str(Euler_angles), (150, 50), 
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("angle", copy)

    # Reassignment
    last_frame = frame
    # kp1 = kp2
    # des1 = des2
    R_old = R_new
