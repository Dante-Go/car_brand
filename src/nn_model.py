import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

data_dir = "/u02/dataset/car_brands/data"

train = True

model_path = "/home/utopa/car_brand_tf/model/image_model"

def convertjpg(jpgfile, width=256, height=256):
	img = Image.open(jpgfile)
	try:
		new_img = img.resize((width, height), Image.BILINEAR)
	except Exception as e:
		print(e)
	return new_img


def load_labels(path):
	car_type_2_labels = {}
	labels_2_car_type = {}
	i = 1
	for label_dir in os.listdir(path):
		labels_2_car_type[i] = label_dir
		car_type_2_labels[label_dir] = i
		i += 1
#	labels_2_car_type[0] = '非车类'
#	car_type_2_labels['非车类'] = 0
	return car_type_2_labels, labels_2_car_type

car_type_2_labels, labels_2_car_type = load_labels(data_dir)


def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for label_dir in os.listdir(data_dir):
        tmp_path = os.path.join(data_dir, label_dir)
        for fname in os.listdir(tmp_path):
            fpath = os.path.join(tmp_path, fname)
            img = convertjpg(fpath)
            fpaths.append(fpath)
            data = np.array(img) / 255.0
            datas.append(data) 
            label = car_type_2_labels[label_dir]
            labels.append(label)
        print(tmp_path)
    return fpaths, datas, labels

fpaths, datas, labels = read_data(data_dir)

num_classes = len(set(labels))

datas_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

dropout_placeholder = tf.placeholder(tf.float32)

conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
flatten = tf.layers.flatten(pool1)
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
dropout_fc = tf.layers.dropout(fc, dropout_placeholder)
logits = tf.layers.dense(dropout_fc, num_classes)
predicted_labels = tf.arg_max(logits, 1)

losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels_placeholder, num_classes), logits=logits)

mean_loss = tf.reduce_mean(losses)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

saver = tf.train.Saver()

with tf.Session() as sess:
	if train:
		print('training')
		sess.run(tf.global_variables_initializer())
		train_feed_dict = {datas_placeholder: datas, labels_placeholder: labels, dropout_placeholder: 0.25 }
		for step in range(150):
			_, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
			if step % 10 == 0:
				print('step = {}\tmean loss = {}'.format(step, mean_loss_val))
		saver.save(sess, model_path)
		print('training done, model path = {}'.format(model_path))
	else:
		print('predict')
		saver.restore(sess, model_path)
		print('load model from {}'.format(model_path))
		label_name_dict = labels_2_car_type
		test_feed_dict = { datas_placeholder: datas, labels_placeholder: labels, dropout_placeholder: 0 }
		predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
		for fpath, real_label, predicted_label in zip(fpaths, predicted_labels_val):
			real_label_name = label_name_dict[real_label]
			predicted_label_name = label_name_dict[predicted_label]
			print('{}\t{} => {}'.format(fpath, real_label_name, predicted_label_name))

# if __name__ == '__main__':
# 	print('car brand')
