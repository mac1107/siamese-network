# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

size = 5  # 图片总体像素大小

train_data = ""
train_label = ""
test_data = ""
test_label = ""
labels = {}
batch = 10
# 定义超参数
lr = 0.001
iter = 200000
filepath='./data/MM.csv'


def init():
    global train_data, train_label, test_data, test_label, labels
    train_data = np.loadtxt(filepath, dtype=np.float, delimiter=',',
                            usecols=(i for i in range(1, 1+size)), skiprows=1)
    train_label = np.loadtxt(filepath, dtype=np.float, delimiter=',',
                             usecols=0, skiprows=1)
    np.random.shuffle(train_data)
    np.random.shuffle(train_label)
    for idx, label in enumerate(train_label):
        if label not in labels:
            labels[label] = [idx]
        else:
            labels[label].append(idx)
    # test_data = np.loadtxt("C:/Users/xucon/Desktop/eval.csv", dtype=np.float, delimiter=',',
    #                        usecols=(i for i in range(1, 10)), skiprows=1)
    # test_label = np.loadtxt("C:/Users/xucon/Desktop/eval.csv", dtype=np.float, delimiter=',',
    #                         usecols=0, skiprows=1)
    # print("test data:{}, test label:{}".format(test_data, test_label))


def get_batch(dataset, labelset):
    idx = np.random.randint(0, len(dataset), size=batch)
    data_batch = []
    label_batch = []
    for i in idx:
        data_batch.append(dataset[i])
        label_batch.append(labelset[i])
    return np.array(data_batch, dtype=float), np.array(label_batch, dtype=float)


# 随机从数据中生成一半正样本，一半负样本
# 每一个样本是2张图片的数据拼接
def next_batch_balance(batch_size, train=True):
    """
    抽取一个batch的数据，然后挑选出其中标签相同的构成正样本，
    再随机挑选出相同数量的作为负样本（其中有可能包含了正样本）
    :param mnist:
    :param batch_size:
    :param train:  获取不同的数据集数据
    :return: data  返回横向叠加的图片数据
    y 就是孪生网络需要的0，1 标签
    """
    if train:
        # x_1, y_1 = mnist.train.next_batch(batch_size)
        # x_2, y_2 = mnist.train.next_batch(batch_size)
        x_1, y_1 = get_batch(train_data, train_label)
        x_2, y_2 = get_batch(train_data, train_label)
    else:
        # x_1, y_1 = mnist.validation.next_batch(batch_size)
        # x_2, y_2 = mnist.validation.next_batch(batch_size)
        x_1, y_1 = get_batch(train_data, train_label)
        x_2, y_2 = get_batch(train_data, train_label)

    temp0 = np.where(y_1 == y_2)[0]
    temp1 = np.random.randint(batch_size, size=len(temp0))
    index = np.union1d(temp0, temp1)
    data = np.hstack((x_1[index, :], x_2[index, :]))

    # y = np.array(y_1 != y_2, dtype=np.float32).reshape(-1,1)[index,:]
    y = np.array(y_1 != y_2, dtype=np.float32)[index]
    return data, y


# 定义可共享网络
def G(x, varReuse, name):
    """
    # 输入是二维的，[batch, one_dim_image(784)] > [batch, 500] > [batch, 10]
    # 这就是在把输入的图片映射到低维度的特征空间了，输出是[batch, 10]，就是图片映射到10维空间
    :param x:
    :param varReuse:
    :param name:
    :return:
    """
    with tf.name_scope(name=name):
        with tf.variable_scope('g_hiden', reuse=varReuse):
            w_hiden = tf.get_variable('weight1', [size, 500],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            b_hiden = tf.get_variable('bias1', [500], initializer=tf.constant_initializer(0.))
            w_output = tf.get_variable('weight2', [500, 10],
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            b_output = tf.get_variable('bias2', [10], initializer=tf.constant_initializer(0.))
        # 隐藏层
        hidden_linear = tf.nn.bias_add(tf.matmul(x, w_hiden), b_hiden, name='hidden_linear')
        hidden_relu = tf.nn.relu(hidden_linear, name='hidden_relu')
        # 输出层
        output_linear = tf.nn.bias_add(tf.matmul(hidden_relu, w_output), b_output, name='output_linear')
        output_relu = tf.nn.relu(output_linear, name='output')

    return output_relu


def _add_gt_image_summary(x1, x2, label=0):
    # 添加图片 summary
    image1 = tf.reshape(x1, shape=[-1, 28, 28, 1])
    image2 = tf.reshape(x2, shape=[-1, 28, 28, 1])
    label1 = tf.reshape(label, shape=[-1, 1, 1, 1])
    image = tf.concat([image1, image2], 1)

    tf.summary.image('GROUND_TRUTH', image)
    tf.summary.image('LABEL', tf.cast(label1, tf.float32))


def Model(x1, x2, label):
    # 定义模型
    Q = tf.constant(5.0, dtype=tf.float32)
    G_x1 = G(x1, varReuse=False, name='left_siamese')
    G_x2 = G(x2, varReuse=True, name='right_siamese')

    with tf.name_scope(name='E_w'):
        # 计算E_w
        E_w = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(G_x1, G_x2)), 1))
        # E_w = tf.reduce_sum(tf.abs(tf.subtract(G_x1, G_x2)), 1)

    with tf.name_scope(name='loss'):
        pos = tf.multiply(tf.multiply(1 - tf.cast(label, tf.float32), 2. / Q), tf.square(E_w))
        neg = tf.multiply(tf.multiply(tf.cast(label, tf.float32), 2 * Q), tf.exp(-2.77 / Q * E_w))
        Loss = tf.reduce_mean(tf.add(pos, neg))

        y_out = tf.to_float(E_w >= 1.0)
        ACC = tf.reduce_mean(tf.to_float(tf.equal(y_out, label)))

    with tf.name_scope(name='Summary'):
        tf.summary.scalar('Loss', Loss)
        tf.summary.scalar('ACC', ACC)
        tf.summary.histogram('E_w', E_w)
        tf.summary.histogram('y_out', y_out)
        tf.summary.histogram('G_x1', G_x1)
        tf.summary.histogram('G_x2', G_x2)
        tf.summary.histogram('pos', pos)
        tf.summary.histogram('neg', neg)
        tf.summary.histogram('label', label)
        # _add_gt_image_summary(x1, x2)

    return Loss, y_out, ACC, E_w


def main():
    # mnist = input_data.read_data_sets(r"data\mnist", one_hot=False)




    # 定义训练占位字节，训练数据入口
    left_input = tf.placeholder(tf.float32, shape=(None, size))
    right_input = tf.placeholder(tf.float32, shape=(None, size))
    y_input = tf.placeholder(tf.float32, shape=None)

    loss, y_out, ACC, E_w = Model(left_input, right_input, y_input)
    # 定义优化器
    opt = tf.train.GradientDescentOptimizer(lr)
    train_op = opt.minimize(loss)
    summary_op = tf.summary.merge_all()

    # 开始训练
    with tf.Session() as sess:
        log_writer = tf.summary.FileWriter('tensorboard/{}'.format('siamese'), sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(iter):
            data, label = next_batch_balance(batch)
            _, loss_train, ACC_train, summary_train ,ew,yo= sess.run([train_op,
                                                                loss,
                                                                ACC,
                                                                summary_op,E_w,y_out],
                                                               feed_dict={left_input: data[:, 0:size],
                                                                          right_input: data[:, size:],
                                                                          y_input: label
                                                                          }
                                                               )
            # 每训练了200次进行1次验证
            if i % 500 == 99:
                # data_val, label_val = next_batch_balance(batch_size, False)
                # loss_val, ACC_val, summary_val, ew, yo = sess.run([loss,
                #                                                    ACC,
                #                                                    summary_op, E_w, y_out],
                #                                                   feed_dict={
                #                                                       left_input: data_val[:, 0:oneImageDataSize],
                #                                                       right_input: data_val[:, oneImageDataSize:],
                #                                                       y_input: label_val
                #                                                   }
                #                                                   )
                print("%03d" % (i + 1), " loss = %.9f" % (loss_train,),
                      'ACC:', ACC_train, 'ACC_train:', ACC_train, 'ew:', ew, 'yo:', yo, 'ys', label)

                log_writer.add_summary(summary_train, i)


if __name__ == "__main__":
    init()
    main()
