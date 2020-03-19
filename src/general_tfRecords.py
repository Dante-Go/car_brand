# coding=utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf

data_dir = "/u02/dataset/car_brands/data"
tfRecords_file_path = "../tfRecords_data/"

def load_labels(path):
    car_type_2_labels = {}
    labels_2_car_type = {}
    i = 1
    for label_dir in os.listdir(path):
        labels_2_car_type[i] = label_dir
        car_type_2_labels[label_dir] = i
        i += 1
    return car_type_2_labels, labels_2_car_type

car_type_2_labels, labels_2_car_type = load_labels(data_dir)

def getTrainList():
    with open('train.txt', 'w') as f:
        for label_dir in os.listdir(data_dir):
            tmp_path = os.path.join(data_dir, label_dir)
            for fname in os.listdir(tmp_path):
                fpath = os.path.join(tmp_path, fname)
                line = fpath + " " + str(car_type_2_labels[label_dir]) + "\n"
                f.write(line)

def load_file(example_list_file):
    lines = np.genfromtxt(example_list_file, delimiter=" ", encoding='utf-8', dtype='U75')
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(int(label))
    return np.asarray(examples), np.asarray(labels), len(lines), len(set(labels))

def convertjpg(jpgfile, width=256, height=256):
#     print(jpgfile)
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
    except Exception as e:
        print(e)
    return new_img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def trans2tfRecord(trainFile, name, output_dir, height=256, width=256):
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    _examples, _labels, example_num, catelogy_num = load_file(trainFile)
    filename = name + '.tfrecords'
    filename = os.path.join(output_dir, filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print("NO{}".format(i))
#         example = example.decode("UTF-8")
        image = convertjpg(example, width, height)
        image_raw = image.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())
    writer.close()
    with open(os.path.join(output_dir,'train_sum.txt'), 'w') as f:
        content = "example_num=" + str(example_num)  + "\n" + "catelogy_num=" + str(catelogy_num) + "\n" 
        f.write(content)


if __name__ == "__main__":
    getTrainList()
    trans2tfRecord('train.txt', 'tf_train', tfRecords_file_path)
    
