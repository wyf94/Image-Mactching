#!/home/wyf/anaconda3/envs/test/bin/python
#!coding=utf-8

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

import message_filters
from sensor_msgs.msg import Image, CameraInfo

import image_matching.test.test as test

'''
@description: 在母图中寻找与子图最为相似的位置，并返回子图缩放比例，匹配位置的左上角坐标，相似程度
@param {*} img_mom 母图
@param {*} img_son 子图
@return {*} match_factor, match_coordinate, max_score
'''
def match_image(img_mom, img_son):
    max_score  = 0.0
    h_mom, w_mom, c_mom = img_mom.shape
    h_son, w_son, c_son = img_son.shape
    
    if(h_mom < h_son or w_mom < w_son):
        print("The size of img_mom > = the size of img_son.")
        return 0

    for i in range(10 ,51):
        # 缩小子图
        tmp_factor = float(i / 100)
        template = cv2.resize(img_son, None, fx=tmp_factor, fy=tmp_factor, interpolation=cv2.INTER_AREA)

        # 相关系数匹配方法：cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(img_mom, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if(max_val > max_score):
            max_score = max_val
            match_factor = tmp_factor
            match_coordinate = max_loc
        
    return match_factor, match_coordinate, max_score

'''
@description: 给图像添加mask，并返回mask图片
@param {*} img 输入图像
@param {*} color 颜色
@param {*} factor 添加的mask比重
@return {*} mask_img
'''
def  img_mask(img, color, factor):
    height, width, channel = img.shape
    zeros = np.zeros((img.shape), dtype=np.uint8)
    zeros_mask = cv2.rectangle(zeros, (0, 0), (width, height),
                color, thickness=-1 ) #thickness=-1 表示矩形框内颜色填充
    mask_img = cv2.addWeighted(img, 1, zeros_mask, factor, 0)
    return mask_img

'''
@description: 将子图嵌入母图(方式：cv2.seamlessClone，优点对子图周边进行平滑处理，达到无缝合成效果)，并返回合成图片
@param {*} img_m 母图
@param {*} img_s 子图
@param {*} factor 子图缩放比例
@param {*} center 子图所在母图中的位置,中心点坐标（宽，高）
@param {*} flags 合成模式: NORMAL_CLONE、MIXED_CLONE和MONOCHROME_TRANSFER
@return {*} img_synthesis 叠加之后的图片
'''
def image_synthesis(img_m, img_s, factor, center, flags):
    # Resize subgraph
    img_s_new = cv2.resize(img_s, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    height_m, width_m, channels_m = img_m.shape
    height_s_new, width_s_new, channels_s_new = img_s_new.shape

    if(center[0] <= height_s_new/2 and center[0] >= height_m - height_s_new/2 and center[1] <= width_s_new/2 and center[1] >= width_m - width_s_new/2):
        print("Error: Misconfiguration of the center point causes the subgraph to exceed the boundary")
        return 0
        
    # Create an all white mask
    mask = np.ones(img_s_new.shape, img_s_new.dtype) * 255 
    # Seamlessly clone src into dst and put the results in output
    img_synthesis = cv2.seamlessClone(img_s_new, img_m, mask, center, flags)
    
    return img_synthesis
    
'''
@description: 将子图嵌入（方式:cv2.addWeighted，优点图像达到透明效果)母图，并返回合成图片
@param {*} img_mom 母图
@param {*} img_son  子图
@param {*} factor 缩放因子
@param {*} left_top 子图所在母图中的位置 左上角坐标（宽，高）
@param {*} alpha 图像叠加权重，即透明度
@return {*}  合成图像
'''
def img_add_roi(img_mom, img_son,factor, left_top, alpha):
    img_son = cv2.resize(img_son, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    h, w, c = img_son.shape

    cropped_roi_img = img_mom[left_top[1]:left_top[1]+h, left_top[0]:left_top[0]+w]  # 裁减母图的一个roi区域，裁剪坐标为[y0:y1, x0:x1]
    img_add = cv2.addWeighted(cropped_roi_img, 1-alpha, img_son, alpha, 0)  # 图像以alpha的权重相加
    img_mom[left_top[1]:left_top[1]+h, left_top[0]:left_top[0]+w] = img_add  # 替换母图中的roi区域

    return img_mom

def callback(img_big, img_small):
    global frame_count, factor, coordinate,flag,video0,video1

    bridge1 = CvBridge()
    bridge2 = CvBridge()
    bridge3 = CvBridge()
    image_big = bridge1.imgmsg_to_cv2(img_big,"bgr8")
    image_small = bridge2.imgmsg_to_cv2(img_small,"bgr8")
    # print("image_small size:",image_small.shape)
    print('frame_count: ', frame_count)

    # 每帧图片写入视频
    video0.write(image_big)
    video1.write(image_small)
    
    if(flag == 0):
        factor, coordinate, score = match_image(image_big, image_small)
        flag = 1
        print('factor: ',factor)
        print('coordinate: ',coordinate)
    else:
        img_small_mask = img_mask(image_small, color = (0, 0, 255), factor = 0.6)
        new_img =  img_add_roi(image_big, img_small_mask, factor, coordinate, 0.3)
        new_img = cv2.resize(new_img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        msg = bridge3.cv2_to_imgmsg(new_img, encoding="bgr8")
        img_pub.publish(msg)
        rate.sleep()

        cv2.imshow("new_img", new_img)
        cv2.waitKey(1)    

    frame_count +=1

if __name__ == '__main__':
    # rospy.init_node('showImage',anonymous = True)/
    rospy.init_node('img_matching_pub', anonymous=True)
    img_pub = rospy.Publisher('/image_matching_publisher', Image, queue_size=10)
    rate = rospy.Rate(25)

    # # 保存长短焦摄像头视频
    # fourcc0= cv2.VideoWriter_fourcc(*'XVID')
    # video0 = cv2.VideoWriter('/media/wyf/C49616A3961695D0/yunfeng.wu/longxing_data/output0.avi', fourcc0, 24, (2448,  2048)) 
    # fourcc1= cv2.VideoWriter_fourcc(*'XVID')
    # video1 = cv2.VideoWriter('/media/wyf/C49616A3961695D0/yunfeng.wu/longxing_data/output1.avi', fourcc1, 24, (2448,  2048))

    frame_count = 0
    flag = 0
    image_sub0= message_filters.Subscriber('/bitcq_camera/image_source1', Image)
    image_sub1 = message_filters.Subscriber('/bitcq_camera/image_source0', Image)
    ts = message_filters.TimeSynchronizer([image_sub0, image_sub1], 15)
    ts.registerCallback(callback)
    rospy.spin()
