import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt  # 画图
import matplotlib.gridspec as gridspec  # 定义网格
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import cv2
import os
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont


##读取tfrecord文件
def read_and_decode11(filename):
    filename_queue = tf.train.string_input_producer([filename])  # 根据文件名生成一个队列
    reader = tf.TFRecordReader()  # 定义一个 reader ，读取下一个 record
    _, serialized_example = reader.read(filename_queue)  # 解析读入的一个record
    features = tf.parse_single_example(serialized_example,
                                       features={"label": tf.FixedLenFeature([], tf.int64),
                                                 "image": tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint16)  # 将字符串解析成图像对应的像素组
    img = tf.reshape(img, [28 * 28 * 1])  # reside图片
    # img = tf.reshape(img,[28,28,1])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features["label"], tf.int32)
    return img, label


##索引tfrecords文件的path
path = "./data/"
imgDMQTrain, labelDMQTrain = read_and_decode11(path + "train.tfrecords")
imgDMQTest, labelDMQTest = read_and_decode11(path + "test.tfrecords")
##将label转为one_hot函数
label_train = tf.one_hot(indices=tf.cast(labelDMQTrain, tf.int32), depth=2)  # 将一个值化为一个概率分布的向量
label_test = tf.one_hot(indices=tf.cast(labelDMQTest, tf.int32), depth=2)

# 随机打乱生成batch
img_batch_train, label_batch_train = tf.train.batch([imgDMQTrain, label_train], batch_size=64, capacity=1000)
img_batch_test, label_batch_test = tf.train.batch([imgDMQTest, label_test], batch_size=40, capacity=40)


def Sensitivity_specificity(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1

    sensitivity = staticity_T[positive_position] / (
                staticity_T[positive_position] + staticity_F[(positive_position + 1) % 2])
    specificity = staticity_T[negative_position] / (
                staticity_T[negative_position] + staticity_F[(negative_position + 1) % 2])
    return sensitivity, specificity


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


sess = tf.InteractiveSession()

W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
x = tf.placeholder(tf.float32, [None, 28 * 28 * 1], name="images")  # 注意：x的shape
y_ = tf.placeholder(tf.float32, [None, 2], name="labels")
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 每张图片有28*28个像素点，28*28=784，None表示任意值
# 将x变成4维向量，28*28为宽*高，-1表示不在意输入图片数目，通道数为1（rgb:3）
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 128, 500])
b_fc1 = bias_variable([500])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([500, 50])
b_fc2 = bias_variable([50])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([50, 2])
b_fc3 = bias_variable([2])
F_score = tf.matmul(h_fc2, W_fc3) + b_fc3
y_conv = tf.nn.softmax(F_score)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=F_score, labels=y_))
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用ADAM最优化方法
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=64, decay_rate=0.98, staircase=True)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step,name="train_step")
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    output_position = tf.argmax(y_conv, 1)
    label_position = tf.argmax(y_, 1)
    predict = tf.equal(output_position, label_position)
    Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
    acc_summary = tf.summary.scalar('Accuracy', Accuracy)
merge = tf.summary.merge_all()

###################################### Grad-cam ################################
cost = (-1) * tf.reduce_sum(tf.multiply(y_, tf.log(F_score)), axis=1)
y_c = tf.reduce_sum(tf.multiply(F_score, y_), axis=1)
target_conv_layer1 = h_conv2
target_conv_layer_grad1 = tf.gradients(y_c, target_conv_layer1)[0]


# gb_grad = tf.gradients(cost, x_image)[0]

def plot(pic, conv_output, conv_grad, step):
    def getcam(image, output, grads_val):
        weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
        cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]
        # Taking a weighted average
        for k, w in enumerate(weights):
            cam += w * output[:, :, k]
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = resize(cam, (28, 28), preserve_range=True)
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        img = image.astype(float)
        img -= np.min(img)
        img /= img.max()

        return img, cam_heatmap

    for i in range(40):  # batch number
        image1 = pic[i, :, :, 0]
        output1 = conv_output[i]
        grads_val1 = conv_grad[i]
        img1, cam1 = getcam(image1, output1, grads_val1)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(image1, cmap=plt.get_cmap('gray'))

        ax = fig.add_subplot(122)
        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(cam1)

        plt.savefig('./image/{}.png'.format(str(step) + '_sample' + str(i), bbox_inches='tight'))


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    #    path2="./log/"
    #    if (not(os.path.exists(path2))):
    #            os.mkdir(path2)
    test_writer = tf.summary.FileWriter(path2 + "test", tf.get_default_graph())
    train_writer = tf.summary.FileWriter(path2 + 'train')

    ##tensorboard的writer
    coord = tf.train.Coordinator()  # 协同启动的线程
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    start_time = time.time()
    f = open(r'./log/XOY.txt', 'w')

    for i in range(10000):
        global_step = i
        img_xs, label_xs = sess.run([img_batch_train, label_batch_train])  # 读取训练 batch
        img_test_xs, label_test_xs = sess.run([img_batch_test, label_batch_test])  # 读取测试batch

        if i % 50 == 0:  # 每一百次迭代输出一次
            img, o, t, summary, acc, output_position_r, label_position_r, predict_r, p = sess.run(
                [x_image, h_conv2, target_conv_layer_grad1, merge, Accuracy, output_position, label_position, predict,
                 y_conv], feed_dict={x: img_test_xs, y_: label_test_xs, keep_prob: 1.0})
            #            test_writer.add_summary(summary,global_step=i)
            sen, spe = Sensitivity_specificity(output_position_r, predict_r)
            fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            s = []
            for k in range(len(p)):
                s.append(p[k][0])
            r, pvalue = stats.pearsonr(s, label_position_r)
            print(' step ' + str(i) + ' ' + str(acc) + ' ' + str(sen) + ' ' + str(spe) + ' ' + str(roc_auc) + ' ' + str(
                pvalue), file=f)
            plot(img, o, t, i)

        if i % 99 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merge, train_step], feed_dict={x: img_xs, y_: label_xs, keep_prob: 0.5},
                                  options=run_options, run_metadata=run_metadata)
        #            train_writer.add_run_metadata(run_metadata,'step%03d' % i)
        #            train_writer.add_summary(summary,i)
        #
        else:
            sess.run([train_step], feed_dict={x: img_xs, y_: label_xs, keep_prob: 0.5})
    ##########################
    print('Total time is: ' + str((time.time() - start_time) / 60) + ' minutes.', file=f)
    coord.request_stop()
    coord.join(threads)
train_writer.close()
test_writer.close()
f.close()
sess.close()
