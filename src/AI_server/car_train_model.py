# coding=utf-8
import tensorflow as tf
import tensorboard

from AI_server.CarNet_v1 import CAR_BRAND_MODEL, img_height, img_width

tfRecords_train_file = "/home/utopa/car_brand_tf/tfRecords_data/tf_train.tfrecords"
train_sum_file = "/home/utopa/car_brand_tf/tfRecords_data/tf_train_sum.txt"

tfRecords_val_file = "/home/utopa/car_brand_tf/tfRecords_data/tf_val.tfrecords"
val_sum_file = "/home/utopa/car_brand_tf/tfRecords_data/tf_val_sum.txt"

tensorboard_dir = "/home/utopa/car_brand_tf/tensorboard_view/view/train_01"

batch = 128
epoch = 20000

def load_sum_info(sum_file):
	with open(sum_file, 'r') as f:
		line = f.readline()
		_, exmaple_num = line.split('=', 1)
		line = f.readline()
		_, category_num = line.split('=', 1)
	return int(exmaple_num), int(category_num)


def read_tfRecord(file_tfRecord, shuffle=False, epochs=None):
	filename_queue = tf.train.string_input_producer([file_tfRecord], shuffle=shuffle, num_epochs=epochs)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
				'image_raw': tf.FixedLenFeature([], tf.string),
				'label': tf.FixedLenFeature([], tf.int64)
			}
		)
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.reshape(image, [img_height, img_width, 3])
	image = tf.cast(image, tf.float32)
	image = tf.image.per_image_standardization(image)
	label = tf.cast(features['label'], tf.int64)
# 	print(image, label)
	return image, label

writer = tf.summary.FileWriter(tensorboard_dir)


with tf.Session() as sess:
	print('training')
	example_num, category_num = load_sum_info(train_sum_file)
# 	img_batch, label_batch = read_tfRecord(tfRecords_train_file, shuffle=True, epochs=epoch)
	img_batch, label_batch = read_tfRecord(tfRecords_train_file, shuffle=True)
	val_img_batch, val_label_batch = read_tfRecord(tfRecords_val_file, shuffle=True)
# 	val_img_batch, val_label_batch = None, None
	min_after_dequeue = 256
	capacity = min_after_dequeue + 3*batch
	image_batches, label_batches = tf.train.shuffle_batch([img_batch, label_batch], batch_size=batch, capacity=capacity, min_after_dequeue=min_after_dequeue)
	val_image_batches, val_label_batches = tf.train.shuffle_batch([val_img_batch, val_label_batch], batch_size=256, capacity=1024, min_after_dequeue=min_after_dequeue)
	model = CAR_BRAND_MODEL(sess=sess, category_n=category_num, example_n=example_num, batch_size=batch, epochs=epoch, tb_writer=writer, is_train=True)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	model.restore_car_model()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	print(threads)
	try:
		if not coord.should_stop():
			model.fit_CAR_BRAND(source_train=image_batches, y_train=label_batches, source_val=val_image_batches, y_val=val_label_batches)
# 			model.fit_CAR_BRAND(source_train=image_batches, y_train=label_batches, source_val=None, y_val=None)
# 			print('test')
	except tf.errors.OutOfRangeError:
		print('Catch OutOfRangeError')
	finally:
		coord.request_stop()
		print('Finished')
	coord.join(threads)
	sess.close()


