#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
# import slim.nets.inception_v3 as inception_v3
# import slim.nets.mobilenet_v1 as mobilenet_v1
import AI_server.TF_slim_models.slim.nets.mobilenet_v1 as mobilenet_v1
from AI_server.TF_slim_models.create_tf_record import *
import tensorflow.contrib.slim as slim

import AI_server.TF_slim_models.global_defines as gbl



# def g_crop_region(x1, y1, x2, y2, img):
#         crop_x1 = x1
#         crop_x2 = x2 
#         crop_y1 = y1
#         crop_y2 = y2
#         crop_w = crop_x2 - crop_x1
#         if crop_w > img.shape[1]:
#             crop_w = img.shape[1]
#         crop_h = crop_y2 - crop_y1
#         if crop_h > img.shape[0]:
#             crop_h = img.shape[0]
#         center_x = int((crop_x1 + crop_x2)/2)
#         center_y = int((crop_y1 + crop_y2)/2)
#         tmp = 0
#         if crop_w > crop_h:
#             tmp = int(crop_w/2)
#         else:
#             tmp = int(crop_h/2)
#         crop_x1 = center_x - tmp
#         crop_x2 = center_x + tmp
#         crop_y1 = y2 - (tmp*2)
#         crop_y2 = y2
#         if crop_x1 < 0:
#             crop_x1 = 0
#         if crop_y1 < 0:
#             crop_y1 = 0
#         if crop_x2 > img.shape[1]:
#             crop_x2 = img.shape[1]
#             crop_x1 = crop_x2 - (tmp*2)
#         if crop_y2 > img.shape[0]:
#             crop_y2 = img.shape[0]
#             crop_y1 = crop_y2 - (tmp*2)
#         return crop_y1, crop_y2, crop_x1, crop_x2
# 
# def g_read_img(image, resize_height,resize_width):
#         im = cv2.imread(image)
#         resize_h = 400
#         height = im.shape[0]
#         scale = im.shape[1] / float(height)
#         img = cv2.resize(im, (int(scale*resize_h), resize_h)) 
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cascade = cv2.CascadeClassifier(gbl.cascade_path)
#         car_plates = cascade.detectMultiScale(img_gray, 1.1, 2, minSize=(36, 9), maxSize=(36*40, 9*40))
#         crop_img = img
# #         print(car_plates)
#         if len(car_plates) > 0:
#             for car_plate in car_plates:
#                 x, y, w, h = car_plate
#                 if w<10 or h < 10:
#                     continue
#                 edge = w
#                 if w < h:
#                     edge = h
#                 crop_x = x 
#                 crop_y = y - edge
#                 crop_x2 = x + edge
#                 crop_y2 = y
#                 y1, y2, x1, x2 = g_crop_region(crop_x, crop_y, crop_x2, crop_y2, img)
#                 crop_img = img[y1:y2, x1:x2]
# #                 str = image.replace('.jpg', 'brand.jpg')
# #                 print(str)
# #                 cv2.imwrite(str, crop_img)
#         rgb_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)#将BGR转为RGB
#         np_image = cv2.resize(rgb_image, (resize_width, resize_height))
#         np_image=np.asanyarray(np_image)
#         np_image=np_image/255.0
# #         np_image=np_image[np.newaxis,:]
#         return np_image
# 
# def  predict(models_path,image_dir,labels_filename,labels_nums, data_format):
#     [batch_size, resize_height, resize_width, depths] = data_format
# 
#     labels = np.loadtxt(labels_filename, str, delimiter='\t')
#     input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 
#     #其他模型预测请修改这里
# #     with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
# #         out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
#     with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
#         out, end_points = mobilenet_v1.mobilenet_v1(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False, global_pool=True)
# 
#     # 将输出结果进行softmax分布,再求最大概率所属类别
#     score = tf.nn.softmax(out,name='pre')
#     class_id = tf.argmax(score, 1)
# 
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
# #     saver.restore(sess, models_path)
#     ckpt = tf.train.get_checkpoint_state(gbl.model_base_path)
#     if ckpt and ckpt.model_checkpoint_path:
#         print(ckpt.model_checkpoint_path)
#         print(ckpt)
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print('Model restored...')
#         
#     images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
#     for image_path in images_list:
#         im=g_read_img(image_path,resize_height,resize_width)
# #         im = read_image(image_path, resize_height, resize_width, normalization=True)
#         im=im[np.newaxis,:]
#         #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
#         pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
#         max_score=pre_score[0,pre_label]
#         print("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score))
#     sess.close()
    


class CarPredict_slim_mobilenet_V1(object):
    def __init__(self, models_path,labels_filename,labels_nums, data_format):
        self._models_path = models_path
        self._labels_file = labels_filename
        self._labels_nums = labels_nums
        self._labels = np.loadtxt(labels_filename, str, delimiter='\t')
        self._dataformat =  data_format
        [batch_size, resize_height, resize_width, depths] = self._dataformat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            self._out, self._end_points = mobilenet_v1.mobilenet_v1(inputs=self.input_images, num_classes=self._labels_nums, dropout_keep_prob=1.0, is_training=False, global_pool=True)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # 将输出结果进行softmax分布,再求最大概率所属类别
        self._score = tf.nn.softmax(self._out,name='pre')
        self._class_id = tf.argmax(self._score, 1)
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        self.restore_car_model()
        self.cascade = cv2.CascadeClassifier(gbl.cascade_path)
        
        
    def __del__(self):
        self.sess.close()
        
    def car_predict(self, image):
        im = self._image_process(image)
        pre_score,pre_label = self.sess.run([self._score,self._class_id], feed_dict={self.input_images:im})
        max_score=pre_score[0,pre_label]
        print("{} is : pre labels:{},name:{} score: {}".format(image, pre_label,self._labels[pre_label], max_score))
        class_code = pre_label
        car_type = self._labels[pre_label][0]
        return class_code[0], car_type
    
    def crop_region(self, x1, y1, x2, y2, img):
        crop_x1 = x1
        crop_x2 = x2 
        crop_y1 = y1
        crop_y2 = y2
        crop_w = crop_x2 - crop_x1
        if crop_w > img.shape[1]:
            crop_w = img.shape[1]
        crop_h = crop_y2 - crop_y1
        if crop_h > img.shape[0]:
            crop_h = img.shape[0]
        center_x = int((crop_x1 + crop_x2)/2)
        center_y = int((crop_y1 + crop_y2)/2)
        tmp = 0
        if crop_w > crop_h:
            tmp = int(crop_w/2)
        else:
            tmp = int(crop_h/2)
        crop_x1 = center_x - tmp
        crop_x2 = center_x + tmp
        crop_y1 = y2 - (tmp*2)
        crop_y2 = y2
        if crop_x1 < 0:
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y1 = 0
        if crop_x2 > img.shape[1]:
            crop_x2 = img.shape[1]
            crop_x1 = crop_x2 - (tmp*2)
        if crop_y2 > img.shape[0]:
            crop_y2 = img.shape[0]
            crop_y1 = crop_y2 - (tmp*2)
        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def _image_process(self, image):
#         im=read_image(image,resize_height,resize_width,normalization=True)
#         im=im[np.newaxis,:]
#         np_image = im
#         np_image = np.asarray(image)
        
#         im = cv2.imread(image)
        im = image
        resize_h = 400
        height = im.shape[0]
        scale = im.shape[1] / float(height)
        img = cv2.resize(im, (int(scale*resize_h), resize_h)) 
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         car_plates = self.cascade.detectMultiScale(img_gray, 1.1, 2, minSize=(36, 9), maxSize=(36*40, 9*40))
#         crop_img = img
#         if len(car_plates) > 0:
#             for car_plate in car_plates:
#                 x, y, w, h = car_plate
#                 if w<10 or h < 10:
#                     continue
#                 crop_x = x 
#                 crop_y = y - int(h*3.5)
#                 crop_x2 = x + w
#                 crop_y2 = y + int(h/2)
#                 y1, y2, x1, x2 = self.crop_region(crop_x, crop_y, crop_x2, crop_y2, img)
#                 crop_img = img[y1:y2, x1:x2]
#         rgb_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)#将BGR转为RGB
#         tmp_name = 'ttt_{}_{}_{}_{}.jpg'.format(y1, y2, x1, x2)
#         cv2.imwrite(tmp_name,rgb_image)
        rgb_image = img
        np_image = cv2.resize(rgb_image, (self.resize_width, self.resize_height))
        np_image=np.asanyarray(np_image)
        np_image=np_image/255.0
        np_image=np_image[np.newaxis,:]
        return np_image
    
    def restore_car_model(self):
        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            print(ckpt)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored...')
    
    

if __name__ == '__main__':

    class_nums=gbl.labels_nums
    image_dir= os.path.join(gbl.dataset_base_path, 'test_image')
#     labels_filename='dataset/label.txt'
    labels_filename = os.path.join(gbl.dataset_base_path, 'labels.txt')
#     models_path='models/model.ckpt-10000'
    models_path = os.path.join(gbl.model_base_path)

    batch_size = 1  #
    resize_height = gbl.resize_height  # 指定存储图片高度
    resize_width = gbl.resize_width  # 指定存储图片宽度
    depths=gbl.depths
    data_format=[batch_size,resize_height,resize_width,depths]
#     predict(models_path,image_dir, labels_filename, class_nums, data_format)

    
    car_predict_model = CarPredict_slim_mobilenet_V1(models_path, labels_filename, class_nums, data_format)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        car_id, car_name = car_predict_model.car_predict(image_path)
#         print(image_path)
#         print(car_id)
#         print(car_name)

        
    
