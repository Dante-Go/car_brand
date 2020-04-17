# coding=utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from AI_server.CarNet_v1 import CAR_BRAND_MODEL

img_width = 256
img_height = 256

train_sum_file = "/home/utopa/car_brand_tf/tfRecords_data/tf_train_sum.txt"
data_base_dir = "/u02/dataset/car_brands/data/"

tensorboard_dir = "/home/utopa/car_brand_tf/tensorboard_view/view/train_01"

def load_labels(path):
    car_type_2_labels = {}
    labels_2_car_type = {}
    i = 0
    for label_dir in os.listdir(path):
        labels_2_car_type[i] = label_dir
        car_type_2_labels[label_dir] = i
        i += 1
    return car_type_2_labels, labels_2_car_type

car_type_2_labels, labels_2_car_type = load_labels(os.path.join(data_base_dir, 'train'))

def convertjpg(jpgfile, width=256, height=256):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
    except Exception as e:
        print(e)
    return new_img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_sum_info(sum_file):
    with open(sum_file, 'r') as f:
        line = f.readline()
        _, exmaple_num = line.split('=', 1)
        line = f.readline()
        _, category_num = line.split('=', 1)
    return int(exmaple_num), int(category_num)



# test_img = '../test/20190926154923_in_v_1_粤AAL249_noBG.jpg'
test_img = '../../../api_test_img.jpg'

def prewhiten(x):
    print(x.size)
    mean = np.mean(x)
    print(mean)
    std = np.std(x)
    print(std) 
    print(np.sqrt(x.size))
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    print(std_adj)
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    print(y.size)
    return y 
# img = convertjpg(test_img)
# # print(img.size)
# img = np.asarray(img)
# data = prewhiten(img)
# data = np.asarray([data, ])
# data = np.reshape(data, (-1, 256,256, 3))

# (256, 256)
# (256, 256)
# 151.3553212483724
# 83.02921600933755
# [16. 16.]
# [83.02921601 83.02921601]


# image = convertjpg(test_img)
# image_raw = image.tobytes()
# example = _bytes_feature(image_raw)
# 
# example = tf.train.Example(features = tf.train.Features(feature={
#                 'image_raw': _bytes_feature(image_raw)
#             }))
# serialized_example = example.SerializeToString()
# features = tf.parse_single_example(
#         serialized_example,
#         features={
#                 'image_raw': tf.FixedLenFeature([], tf.string)
#             }
#         )
# image = tf.decode_raw(features['image_raw'], tf.uint8)
# image = tf.reshape(image, [img_width, img_height, 3])
# image = tf.cast(image, tf.float32)
# image = tf.image.per_image_standardization(image)
# data = tf.reshape(image, [-1, img_width, img_height, 3])


# writer = tf.summary.FileWriter(tensorboard_dir)
 
# with tf.Session() as sess:
#     print('predict')
#     example_num, category_num = load_sum_info(train_sum_file)
#     model = CAR_BRAND_MODEL(sess=sess, category_n=category_num, example_n=None, batch_size=None, epochs=None, tb_writer=writer, is_train=False)
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init_op)
#     model.restore_car_model()
#     classes_code = model.predict(data)
#     print(classes_code)
#     car_type = labels_2_car_type[classes_code[0]]
#     print(car_type)
#     print(labels_2_car_type)
#     sess.close()


class CarPredict(object):
    def __init__(self):
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(tensorboard_dir)
        example_num, category_num = load_sum_info(train_sum_file)
        self.model = CAR_BRAND_MODEL(sess=self.sess, category_n=category_num, example_n=None, batch_size=None, epochs=None, tb_writer=self.writer, is_train=False)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.model.restore_car_model()
        
    def __del__(self):
        self.sess.close()
        
    def car_predict(self, image):
        data = self._image_process(image)
        class_code = self.model.predict(data)
        car_type = labels_2_car_type[class_code[0]]
#         car_type = ''
        return class_code[0], car_type
    
    def _image_process(self, image):
        image = convertjpg(image)
#         image_raw = image.tobytes()
#         example = tf.train.Example(features = tf.train.Features(feature={
#                         'image_raw': _bytes_feature(image_raw)
#                     }))
#         serialized_example = example.SerializeToString()
#         features = tf.parse_single_example(
#                 serialized_example,
#                 features={
#                         'image_raw': tf.FixedLenFeature([], tf.string)
#                     }
#                 )
#         image = tf.decode_raw(features['image_raw'], tf.uint8)
#         image = tf.reshape(image, [img_width, img_height, 3])
#         image = tf.cast(image, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         data = tf.reshape(image, [-1, img_width, img_height, 3])
#         image_raw = image.flatten()
#         print(image_raw.size)
        np_image = np.asarray(image)
#         np_image = np_image.flatten()
        print(np_image.size)
        np_image = np_image.astype(float)
        data = prewhiten(np_image)
        data = np.reshape(data, [-1, img_width, img_height, 3])
        return data
        

# if __name__ == '__main__':
#     car_pre = CarPredict()
#     class_code, car_type = car_pre.predict(test_img)
#     print(class_code)
#     print(car_type)
        