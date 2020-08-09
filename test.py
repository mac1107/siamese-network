import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from siamese import *

# tf.enable_eager_execution()
tf.reset_default_graph()

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)


# print(mnist.validation.num_examples)
# print(mnist.train.num_examples)
# print(mnist.test.num_examples)
# print(mnist.train)


def get_dataset(file_path):
    # CSV_COLUMNS = ['label', 'asp1', 'asp2', 'asp3', 'asp4', 'asp5', 'asp6', 'asp7', 'asp8', 'asp9', 'asp10', 'asp11',
    #                'asp12', 'asp13', 'asp14', 'asp15', 'asp16', 'asp17', 'asp18', 'asp19', 'asp20', 'haus1', 'hasu2',
    #                'haus3', 'haus4', 'haus5']
    LABEL_COLUMN = 'survived'
    LABELS = [0, 1]
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,  # 为了示例更容易展示，手动设置较小的值
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


def print_dataset(data_set):
    iterator = data_set.make_one_shot_iterator()
    next_element = iterator.get_next()
    num_batch = 0
    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            value = sess.run(next_element)
            num_batch += 1
            print("Num Batch: ", num_batch)
            print("Batch value: ", value)


train_data = ""
train_label = ""


def init():
    global train_data, train_label
    train_data = np.loadtxt("/Users/bytedance/.keras/datasets/trainfuck.csv", dtype=np.float, delimiter=',',
                            usecols=(i for i in range(1, 10)))
    train_label = np.loadtxt("/Users/bytedance/.keras/datasets/trainfuck.csv", dtype=np.float, delimiter=',',
                             usecols=0)
    print(train_data)


def get_batch(batch_size):
    idx = np.random.randint(0, len(train_data), size=batch_size)
    data_batch = []
    label_batch = []
    for i in idx:
        data_batch.append(train_data[i])
        label_batch.append(train_label[i])
    return np.array(data_batch, dtype=float), np.array(label_batch, dtype=float)


def run():
    lr = 0.01
    iterations = 20000
    batch_size = 16

    # 占位符
    with tf.variable_scope('input_x1') as scope:
        # x1 = tf.placeholder(tf.float32, shape=[None, 784])
        # x_input_1 = tf.reshape(x1, [-1, 28, 28, 1])
        # x1 = tf.placeholder(tf.float32, shape=[None, 784])
        # x_input_1 = tf.reshape(x1, [-1, 28, 28, 1])
        x1 = tf.placeholder(tf.float32, shape=[None, 9])
        x_input_1 = tf.reshape(x1, [-1, 3, 3, 1])
    with tf.variable_scope('input_x2') as scope:
        # x2 = tf.placeholder(tf.float32, shape=[None, 784])
        # x_input_2 = tf.reshape(x2, [-1, 28, 28, 1])
        # x2 = tf.placeholder(tf.float32, shape=[None, 784])
        # x_input_2 = tf.reshape(x1, [-1, 28, 28, 1])
        x2 = tf.placeholder(tf.float32, shape=[None, 9])
        x_input_2 = tf.reshape(x1, [-1, 3, 3, 1])
    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape=[batch_size])

    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('siamese') as scope:
        out1 = siamese(x_input_1, keep_prob)
        scope.reuse_variables()
        out2 = siamese(x_input_2, keep_prob)
    with tf.variable_scope('metrics') as scope:
        loss = siamese_loss(out1, out2, y)
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    loss_summary = tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()
    # iterator = raw_train_data.make_one_shot_iterator()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graph/siamese', sess.graph)
        sess.run(tf.global_variables_initializer())
        # sess.run(iterator.initializer, feed_dict={batch_size: 20})
        for itera in range(iterations):
            xs_1, ys_1 = get_batch(16)
            # xs_1, ys_1 = mnist.train.next_batch(batch_size)
            # print(type(xs_1), xs_1, len(xs_1))
            # print(type(ys_1), ys_1, len(ys_1))
            # xs_1,ys_1=raw_train_data.batch(2)
            # ys_1 = np.argmax(ys_1, axis=1)
            xs_2, ys_2 = get_batch(16)
            # xs_2, ys_2 = mnist.train.next_batch(batch_size)
            # ys_2 = np.argmax(ys_2, axis=1)
            # print(type(ys_1), ys_2)
            y_s = np.array(ys_1 == ys_2, dtype=np.float32)
            _, train_loss, summ = sess.run([optimizer, loss, merged_summary],
                                           feed_dict={x1: xs_1, x2: xs_2, y: y_s, keep_prob: 0.6})

            writer.add_summary(summ, itera)
            if itera % 1000 == 1:
                print('iter {},train loss {}'.format(itera, train_loss))
        # embed = sess.run(out1, feed_dict={x1: mnist.test.images, keep_prob: 0.6})
        # test_img = mnist.test.images.reshape([-1, 3, 3, 1])
        writer.close()


init()

run()


def test():
    a = [1, 2, 3, 4]
    print(np.array(a), type(np.array(a)))


test()
