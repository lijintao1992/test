import cv2
import dlib
import numpy as np
import os
import os.path

video_capture=cv2.VideoCapture(0)
face_detector=dlib.get_frontal_face_detector()

current_directory=os.path.dirname(os.path.realpath(__file__))
landmark_file_name='shape_predictor_68_face_landmarks.dat'
# full_path=os.path.realpath(os.path.join(current_directory,'./'+landmark_file_name))
# print(full_path)
full_path=os.path.realpath(os.path.join(current_directory,landmark_file_name))
# print(os.path.realpath(os.path.join(current_directory,landmark_file_name)))
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# print(current_directory)
sun_glasses_path=os.path.realpath(os.path.join(current_directory,'.\images\sunglass.jpg'))


def trace_human_gray(image):
    return image


def rect_to_bound_box(rect):
    x1=rect.left()
    y1=rect.top()
    x2=rect.right()
    y2=rect.bottom()
    return x1,y1,x2,y2

def scale_back(bounding_box,scale):
    (x1, y1, x2, y2)=bounding_box
    x1 = int(x1/scale)
    y1 = int(y1 / scale)
    x2 = int(x2 / scale)
    y2 = int(y2 / scale)
    return x1,y1,x2,y2

def scale_point_back(point,scale):
    (x,y)= point
    x= int(x / scale)
    y= int(y / scale)
    return x,y

def shape_to_np(shape):
    coords=np.zeros((68,2),'int')
    for i in range(68):
        coords[i]=(shape.part(i).x, shape.part(i).y)
    return coords

def add_sungass(image,center_point,width):
    sun_glass = cv2.imread('./images/sunglass.jpg')
    height=int(sun_glass.shape[0]/sun_glass.shape[1]*width)
    center_x,center_y=center_point
    start_x=int(center_x-width/2)
    start_y=int(center_y-height/2)
	bbb= 111
	aaa= 222
    sun_glass = cv2.resize(sun_glass, (width,height), interpolation=cv2.INTER_AREA)
    image[start_y:start_y+height,start_x:start_x+width]=sun_glass

while True: # 此行确保是活动窗口，不断调用
    ret,image=video_capture.read()
    # thumb表示缩略图,横向竖向都缩放
    scale=200/min(image.shape[0],image.shape[1])
    thumb=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    # 灰度图
    gray=cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY)
    trace_human_gray(gray)
    face_rects=face_detector(gray,1)
    # print(face_rects[0])
    for i, rect in enumerate(face_rects):
        x1, y1, x2, y2=rect_to_bound_box(rect)
        x1, y1, x2, y2=scale_back((x1, y1, x2, y2),scale)
        shape = predictor(gray, rect)
        shape=shape_to_np(shape)
        for point in shape:
            cv2.circle(image,scale_point_back(point,scale),2,(0,0,255),1) # 2指半径
        cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),2)

        # cv2.rectangle(gray, 50, 80, 70, 40, (0, 255, 0),2)
        # print("第", i + 1, "个人脸d的坐标：","left:", x1,"right:", x2,"top:", y1,"bottom:",y2)
        width=int(abs(shape[17][0]-shape[26][0])/scale)

        add_sungass(image, scale_point_back((shape[27][0], shape[27][1]), scale), width+30)
        cv2.imshow('Example',image)
# 实现按q键退出
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()

### 来看看改了什么



