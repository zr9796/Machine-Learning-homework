# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(1):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            print(i)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 定义评价指标
    predictions = tf.argmax(y_conv, 1)
    actuals = tf.argmax(y_, 1)
 
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
 
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )
 
    tn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )
 
    fp_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
    )
 
    fn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )
    # 输出测试集的评价结果
    tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    recall = float(tp)/(float(tp) + float(fn))
    precision = float(tp)/(float(tp) + float(fp))
    F1 = (2 * (precision * recall)) / (precision + recall)
    
    print("test accuracy %g" % accuracy)
    print("test precision %g" % precision)
    print("test recall %g" % recall)
    print("test F1 %g" % F1)
