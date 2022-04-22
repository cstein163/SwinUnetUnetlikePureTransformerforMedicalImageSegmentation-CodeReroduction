#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2022/3/12 21:40
@Author : "ChengZhuoZhi"
@Email : 1521265074@qq.com
@File : getPicFromVideo.py
@Software: PyCharm
'''

import cv2
import os
import pdb
import numpy as np
from glob2 import glob
'''
输入
    # 每隔n帧保存一张图片
    frame_interval = 100
    videos_src_path = ？
    
'''


videos_src_path = 'D:\\StudyMaterials\\FireLineRecongize\\100MEDIA\\100MEDIA\\'  # 提取图片的视频文件夹

# 筛选文件夹下MP4格式的文件
# videos = os.listdir(videos_src_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
# videos = filter(lambda x: x.endswith('mp4'), videos)
dirs = os.listdir(videos_src_path)  # 获取指定路径下的文件


count = 0

# 数总帧数
# total_frame = 0

# 写入txt
f = "D:/StudyMaterials/FireLineRecongize/UnetCode/use_swinunet-pytorch/picFromVideo/log.txt"    # 存放抽取各个视频名称的记录
with open(f, "w+") as file:
    file.write("-----start-----\n")

# 循环读取路径下的文件并操作
for video_name in dirs:

    outputPath = "D:/StudyMaterials/FireLineRecongize/UnetCode/use_swinunet-pytorch/picFromVideo/"       # 存放抽帧图片的路径

    ##生成文件名对应的文件夹，并去掉文件格式后缀
    # outputPath = "C:\\Users\\Zhang\\Desktop\\aaa\\ff_img\\" + name
    # videoNameNoType = video_name.split('.')
    # videoNameNoType = videoNameNoType[0]
    picFromVideoDirPath = outputPath+video_name         # 按照视频名称创建新文件夹的地址
    picFromVideoDir = os.listdir(outputPath)
    '''
    在该目录下新建一个以视频命名的目录，并将抽出来的图片放进这个目录文件里面
    '''
    if video_name not in picFromVideoDir:
        # 如果视频文件夹不存在，则创建
        os.mkdir(picFromVideoDirPath)
    print("start\n")

    vc = cv2.VideoCapture(videos_src_path + video_name)

    # 初始化,并读取第一帧
    # rval表示是否成功获取帧
    # frame是捕获到的图像
    rval, frame = vc.read()

    # 获取视频fps
    fps = vc.get(cv2.CAP_PROP_FPS)
    # 获取每个视频帧数
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    # 获取所有视频总帧数
    # total_frame+=frame_all

    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    # print("[INFO] 所有视频总帧: ",total_frame)
    # print("[INFO] 视频时长: {}s".format(frame_all/fps))

    # if os.path.exists(outputPath) is False:
    #     print("[INFO] 创建文件夹,用于保存提取的帧")
    #     os.mkdir(outputPath)

    # 每隔n帧保存一张图片
    frame_interval = 50
    # 统计当前帧
    frame_count = 1

    while rval:

        rval, frame = vc.read()

        # 隔n帧保存一张图片
        if frame_count % frame_interval == 0:

            # 当前帧不为None，能读取到图片时
            if frame is not None:
                # print(picFromVideoDirPath)
                filename = picFromVideoDirPath+"/{}.jpg".format(count)
                # print(filename)

                # # 水平、垂直翻转
                # frame = cv2.flip(frame, 0)
                # frame = cv2.flip(frame, 1)
                #
                # # 旋转90°
                # frame = np.rot90(frame)
                cv2.imwrite(filename, frame)
                count += 1
                print("保存图片:{}".format(filename))
        frame_count += 1

    # 将成功抽帧的视频名称写入txt文件，方便检查
    file = open(f, "a")
    file.write(video_name + "\n")

    # 关闭视频文件
    vc.release()
    print("[INFO] 总共保存：{}张图片\n".format(count))
