# coding=utf-8
import tensorflow as tf
import time
from datetime import timedelta
import os
from AI_server.mobileNetV3_layers import *

model_path = "/home/utopa/car_brand_tf/model/image_model/"

img_width = 256
img_height = 256


class CAR_BRAND_MODEL:
    def __init__(self, sess, category_n, example_n, batch_size, epochs, tb_writer, is_train):
        self._sess = sess
        self._category_num = category_n
        self._example_num = example_n
        self._logits = None
        self._loss = None
        self._mean_loss = None
        self._accuracy = None
        self._optimizer = None
        self._prec = None
        self._saver = None
        self._ckpt = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._tb_writer = tb_writer
        self._summ = None
        
        self.create_model(is_train)
        
        
    def fit_CAR_BRAND(self, source_train, y_train, source_val, y_val):
        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improment = 2000000
        flag = False
        for epoch in range(self._epochs):
            for batch_train in range(self._example_num // self._batch_size):
                X_batch, Y_batch = self._sess.run([source_train, y_train])
#                 print(X_batch.shape)
                feed_dict = {self.X: X_batch, self.y: Y_batch}
                _, train_loss = self._sess.run([self._optimizer, self._mean_loss], feed_dict=feed_dict)
                total_batch += 1
                
            X_val, y_v = self._sess.run([source_val, y_val])
#             losses, acc_val, summ = self._sess.run([self._mean_loss, self._accuracy, self._summ], feed_dict={self.X: X_val, self.y: y_v})
            acc_val, summ = self._sess.run([self._accuracy, self._summ], feed_dict={self.X: X_val, self.y: y_v})
#             acc_val, summ = self._sess.run([self._accuracy, self._summ], feed_dict={self.X: X_batch, self.y: Y_batch})
#             losses = self._sess.run([self._mean_loss], feed_dict={self.X: X_batch, self.y: Y_batch})
#             acc_val = 0
            time_dif = timedelta(seconds=int(round(time.time() - start_time)))
            msg = "Epoch: {0:>4}, Iter: {1:>9}, Time: {2}, loss: {3:>20}, acc: {4}"
            print(msg.format(epoch, total_batch, time_dif, train_loss, acc_val))
            self._tb_writer.add_summary(summ, total_batch)
            if epoch % 10 == 0:
                self.save_car_model(epoch)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                last_improved = total_batch
                print("improved!")
                self.save_car_model(epoch, better=str(best_acc_val))
            if total_batch - last_improved > require_improment:
                print("Early stopping in ", total_batch, " step! And the best validation accuracy is ", best_acc_val, '.')
                flag = True
                break
            if flag:
                break
#             self.save_car_model(epoch)

    def create_model(self, is_train):
        self.X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 3], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self._logits, prec = mobilenetV3_small(self.X, self._category_num, is_train=is_train)
        with tf.name_scope('loss'):
            self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, self._category_num), logits=self._logits)
            self._mean_loss = tf.reduce_mean(self._loss, name="loss")
            tf.summary.scalar('loss', self._mean_loss)
        with tf.name_scope('train'):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self._loss)
        with tf.name_scope('accuracy'):
#             self._accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self._logits, self.y, 1), tf.float32))
            accs = tf.equal(tf.argmax(self._logits, 1), tf.argmax(tf.one_hot(self.y, self._category_num), 1))
            self._accuracy = tf.reduce_mean(tf.cast(accs, tf.float32))
            tf.summary.scalar('accuracy', self._accuracy)
        with tf.name_scope('predict'):
            self._prec = prec 
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        self._summ = tf.summary.merge_all()
        self._tb_writer.add_graph(self._sess.graph)
    
    def save_car_model(self, step, better=''):
        if self._saver == None:
            return
        if better is not '':
            model_file = os.path.join(model_path, "car_model_"+better)
        else:
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
        
    def predict(self, data):
        feed_data = self._sess.run(data)
        feed_dict = {self.X:feed_data}
        class_code = self._sess.run(self._prec, feed_dict=feed_dict)
        return class_code
        
        