# coding=utf-8
import tensorflow as tf
import time
from datetime import timedelta
import os

model_path = "../model/image_model/"

class CAR_BRAND_MODEL:
    def __init__(self, sess, category_n, example_n, batch_size, epochs, tb_writer):
        self._sess = sess
        self._category_num = category_n
        self._example_num = example_n
        self._model = None
        self._loss = None
        self._mean_loss = None
        self._accuracy = None
        self._optimizer = None
        self._saver = None
        self._ckpt = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._tb_writer = tb_writer
        self._summ = None
        
        self.create_model()
        
        
    def fit_CAR_BRAND(self, source_train, y_train, source_val, y_val):
        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improment = 20000000
        flag = False
        for epoch in range(self._epochs):
            for batch_train in range(self._example_num // self._batch_size):
                X_batch, Y_batch = self._sess.run([source_train, y_train])
#                 print(X_batch.shape)
                feed_dict = {self.X: X_batch, self.y: Y_batch}
                self._sess.run(self._optimizer, feed_dict=feed_dict)
                if total_batch % 1000 == 0:
                    X_val, y_v = self._sess.run([source_val, y_val])
#                     losses, acc_val = self._sess.run([self._mean_loss, self._accuracy, self._summ], feed_dict={self.X: X_val, self.y: y_v})
                    losses, acc_val, summ = self._sess.run([self._mean_loss, self._accuracy, self._summ], feed_dict={self.X: X_batch, self.y: Y_batch})
                    time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                    msg = "Epoch: {0:>5}, Iter: {1:>12}, Time: {2}, loss: {3}, acc: {4}"
                    print(msg.format(epoch, total_batch, time_dif, losses, acc_val))
                    self._tb_writer.add_summary(summ, total_batch)
#                     self.save_car_model(epoch)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    print("improved!\n")
                    self.save_car_model(epoch)
                if total_batch - last_improved > require_improment:
                    print("Early stopping in ", total_batch, " step! And the best validation accuracy is ", best_acc_val, '.')
                    flag = True
                    break
                total_batch += 1
            if flag:
                break
            self.save_car_model(epoch)

    def create_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        conv0 = tf.layers.conv2d(self.X, filters=20, kernel_size=5, activation=tf.nn.relu, name="conv_0")
        print(conv0.shape)
        pool0 = tf.layers.max_pooling2d(conv0, pool_size=[2, 2], strides=[2, 2], name="pool_0")
        print(pool0.shape)
        conv1 = tf.layers.conv2d(pool0, filters=40, kernel_size=5, activation=tf.nn.relu, name="conv_1")
        print(conv1.shape)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], name="pool_1")
        conv2 = tf.layers.conv2d(pool1, filters=80, kernel_size=5, activation=tf.nn.relu, name="conv_2")
        print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], name="pool_2")
        conv3 = tf.layers.conv2d(pool2, filters=160, kernel_size=5, activation=tf.nn.relu, name="conv_3")
        print(conv3.shape)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], name="pool_3")
        conv4 = tf.layers.conv2d(pool3, filters=320, kernel_size=5, activation=tf.nn.relu, name="conv_4")
        print(conv4.shape)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[2, 2], name="pool_4")
        flatten = tf.layers.flatten(pool4)
        fc = tf.layers.dense(flatten, units=4000, activation=tf.nn.relu)
        dropout_fc = tf.layers.dropout(fc, tf.float32)
        self._model = tf.layers.dense(dropout_fc, self._category_num)
        with tf.name_scope('loss'):
            self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, self._category_num), logits=self._model)
            self._mean_loss = tf.reduce_mean(self._loss, name="loss")
            tf.summary.scalar('loss', self._mean_loss)
        with tf.name_scope('train'):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
        with tf.name_scope('accuracy'):
            self._accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self._model, self.y, 1), tf.float32))
            tf.summary.scalar('accuracy', self._accuracy)
        if self._saver is None:
                self._saver = tf.train.Saver(max_to_keep=1)
        self._summ = tf.summary.merge_all()
        self._tb_writer.add_graph(self._sess.graph)
    
    def save_car_model(self, step):
        if self._saver == None:
            return
        model_file = os.path.join(model_path, "car_model")
        self._saver.save(self._sess, model_file, global_step=step)
    
    def restore_car_model(self):
        if self._saver == None:
            return 
        self._ckpt = tf.train.get_checkpoint_state(model_path)
        if self._ckpt and self._ckpt.model_checkpoint_path:
            print(self._ckpt.model_checkpoint_path)
            print(self._ckpt)
            self._saver.restore(self._sess, self._ckpt.model_checkpoint_path)
            print('Model restored...')
        
        
        