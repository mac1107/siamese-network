import tensorflow as tf
import numpy as np

train_data = ""
train_label = ""
test_data = ""
test_label = ""
labels = {}
batch = 5


def init():
    global train_data, train_label, test_data, test_label, labels
    train_data = np.loadtxt("./data/temp.csv", dtype=np.float, delimiter=',',
                            usecols=(i for i in range(1, 6)), skiprows=1)
    train_label = np.loadtxt("./data/temp.csv", dtype=np.float, delimiter=',',
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


def siamese_loss(out1, out2, y, Q=1.5):
    Q = tf.constant(Q, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1, out2)), 1))
    pos = tf.multiply(tf.multiply(1 - tf.cast(y, tf.float32), 2. / Q), tf.square(E_w))
    neg = tf.multiply(tf.multiply(tf.cast(y, tf.float32), 2 * Q), tf.exp(-2.77 / Q * E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss, name="loss")
    y_out = tf.to_float(E_w >= 0.75)
    ACC = tf.reduce_mean(tf.to_float(tf.equal(y_out, y)))

    return loss, y_out, ACC, E_w
    # return loss


def siamese(inputs, keep_prob):
    # conv1+relu1

    with tf.name_scope('conv1'):
        w1 = tf.Variable(tf.truncated_normal(shape=[2, 1, 16], stddev=0.01), name='w1')
        # b1 = tf.Variable(tf.zeros(32), name='b1')
        conv1 = tf.nn.conv1d(inputs, w1, 1, padding='SAME', name='conv1')
        # pool1 = tf.nn.max_pool(conv1, ksize=2,strides=1, padding='SAME')
    with tf.name_scope('relu1'):
        relu1 = tf.nn.relu(conv1, name='relu1')
    # conv2+relu2
    with tf.name_scope('conv2'):
        w2 = tf.Variable(tf.truncated_normal(shape=[2, 16, 32], stddev=0.01), name='w2')
        # b2 = tf.Variable(tf.zeros(64), name='b2')
        conv2 = tf.nn.conv1d(relu1, w2, 2, padding='SAME', name='conv2')
        # pool2 = tf.nn.max_pool(conv2, ksize=2,strides=2, padding='SAME')
    with tf.name_scope('relu2'):
        relu2 = tf.nn.relu(conv2, name='relu2')

    # conv3+relu3
    # with tf.name_scope('conv3'):
    #     w3 = tf.Variable(tf.truncated_normal(shape=[2, 16, 32], mean=0, stddev=0.01), name='w3')
    #     conv3 = tf.nn.conv1d(relu2, w3, 2, padding='SAME')
    #     # pool3 = tf.nn.max_pool(conv3, ksize=2,strides=2, padding='SAME')
    # with tf.name_scope('relu3') as scope:
    #     relu3 = tf.nn.relu(conv3, name='relu3')

    # full connect+relu
    with tf.name_scope('fc1') as scope:
        x_flat = tf.reshape(relu2, shape=[-1, 3 * 32])
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[3 * 32, 128], stddev=0.01, mean=0), name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(128), name='b_fc1')
        fc1 = tf.add(tf.matmul(x_flat, w_fc1), b_fc1)
    with tf.name_scope('relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')

    # drop+f
    with tf.name_scope('drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=keep_prob, name='drop_1')
    with tf.name_scope('bn_fc1') as scope:
        bn_fc1 = tf.layers.batch_normalization(drop_1, name='bn_fc1')
    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[128, 2], stddev=0.05, mean=0), name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(2, name='b_fc2'))
        fc2 = tf.add(tf.matmul(bn_fc1, w_fc2), b_fc2)
    with tf.name_scope('relu_fc2') as scope:
        relu_fc2 = tf.nn.relu(fc2, name='relu_fc2')
    # with tf.name_scope('drop_2') as scope:
    #     drop_2 = tf.nn.dropout(relu_fc2, keep_prob=keep_prob, name='drop_2')
    # with tf.name_scope('bn_fc2') as scope:
    #     bn_fc2 = tf.layers.batch_normalization(drop_2, name='bn_fc2')
    # with tf.name_scope('fc3') as scope:
    #     w_fc3 = tf.Variable(tf.truncated_normal(shape=[32, 5], stddev=0.05, mean=0), name='w_fc3')
    #     b_fc3 = tf.Variable(tf.zeros(5), name='b_fc3')
    #     fc3 = tf.add(tf.matmul(bn_fc2, w_fc3), b_fc3)
    return relu_fc2


def run():
    lr = 0.001
    iterations = 10000

    # 占位符
    with tf.variable_scope('input_x1'):

        x1 = tf.placeholder(tf.float32, shape=[None, 5])
        x_input_1 = tf.reshape(x1, [-1, 5, 1])
    with tf.variable_scope('input_x2'):
        x2 = tf.placeholder(tf.float32, shape=[None, 5])
        x_input_2 = tf.reshape(x2, [-1,5, 1])
    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape=[None])

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.variable_scope('siamese') as scope:
        out1 = siamese(x_input_1, keep_prob)
        scope.reuse_variables()
        out2 = siamese(x_input_2, keep_prob)
    with tf.variable_scope('metrics') as scope:
        loss, y_out, ACC, E_w = siamese_loss(out1, out2, y)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('ACC', ACC)
    # tf.summary.histogram('E_w', E_w)
    tf.summary.histogram('y_out', y_out)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graph/siamese', sess.graph)
        sess.run(tf.global_variables_initializer())
        for itera in range(iterations):
            xs_1, ys_1 = get_batch(train_data, train_label)

            xs_2, ys_2 = get_batch(train_data, train_label)

            y_s = np.array(ys_1 != ys_2, dtype=np.float32)
            _, yo, train_loss, summ,ew = sess.run([optimizer, y_out, loss, merged_summary,E_w],
                                               feed_dict={x1: xs_1, x2: xs_2, y: y_s, keep_prob: 0.5,
                                                          })

            writer.add_summary(summ, itera)
            if itera % 1000 == 1:
                print('iter {},train loss {} ys {} y_out {} e_w {}'.format(itera, train_loss, y_s, yo,ew))

        writer.close()


if __name__ == "__main__":
    tf.reset_default_graph()
    init()
    run()
