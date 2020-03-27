# coding=utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from CarNet_v1 import CAR_BRAND_MODEL
from general_tfRecords import convertjpg, load_labels

train_sum_file = "../tfRecords_data/tf_train_sum.txt"
data_base_dir = "../dataset/car_brands/data/"

car_type_2_labels, labels_2_car_type = load_labels(os.path.join(data_base_dir, 'train'))

def load_sum_info(sum_file):
    with open(sum_file, 'r') as f:
        line = f.readline()
        _, exmaple_num = line.split('=', 1)
        line = f.readline()
        _, category_num = line.split('=', 1)
    return int(exmaple_num), int(category_num)


test_img = '../test/test.jpg'

img = convertjpg(test_img)

data = img.tobytes()

with tf.Session() as sess:
    print('predict')
    example_num, category_num = load_sum_info(train_sum_file)
    model = CAR_BRAND_MODEL(sess=sess, category_n=category_num, example_n=None, batch_size=None, epochs=None, tb_writer=None)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    model.restore_car_model()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(threads)
    try:
        if not coord.should_stop():
            classes_code = model.predict(data)
            car_type = labels_2_car_type[classes_code]
            print(car_type)
    except tf.errors.OutOfRangeError:
        print('Catch OutOfRangeError')
    finally:
        coord.request_stop()
        print('Finished')
    coord.join(threads)
    sess.close()


# if __name__ == '__main__':
#     print('car brand')