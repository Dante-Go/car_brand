#coding=utf-8
dataset_base_path = '/u02/dataset/car_brands/data'
# dataset_base_path = '/data/car_brand_model/data'
model_base_path = '/home/utopa/car_brand_tf/model/image_model'
# model_base_path = '/data/car_brand_model/car_brand_tf/model/image_model'

cascade_path = '/home/utopa/car_brand_tf/src/AI_server/TF_slim_models/cascade.xml'

labels_nums = 3  # 类别个数
batch_size = 32  #
resize_height = 128  # mobilenet_v1.default_image_size 指定存储图片高度
resize_width = 128   # mobilenet_v1.default_image_size 指定存储图片宽度
depths = 3

# import os

# data_base_dir = dataset_base_path
# def load_labels(path):
#     car_type_2_labels = {}
#     labels_2_car_type = {}
#     i = 0
#     for label_dir in os.listdir(path):
#         labels_2_car_type[i] = label_dir
#         car_type_2_labels[label_dir] = i
#         i += 1
#     return car_type_2_labels, labels_2_car_type
# car_type_2_labels, labels_2_car_type = load_labels(os.path.join(data_base_dir, 'train'))
# def create_labels_file(path):
#     with open(path, 'w') as f:
#         str = ''
#         for i in range(len(labels_2_car_type)):
#             tmp_str = '{0}\t'.format(labels_2_car_type[i])
#             str += tmp_str
#         f.write(str)
# create_labels_file(os.path.join(data_base_dir, 'labels.txt'))