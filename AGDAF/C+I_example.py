import tensorflow as tf
from scipy.linalg import orth
import time
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os

i = '60'


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={"label": tf.FixedLenFeature([], tf.int64),
                                                     "image": tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["image"], tf.int16)
        img = tf.reshape(img, [28 * 28 * 1])
        max = tf.to_float(tf.reduce_max(img))
        img = tf.cast(img, tf.float32) * (1.0 / max)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size, capacity)
    return data, label


batch_size = 40

DMQ_Axial_dataTrain, DMQ_Axial_labelTrain = get_inputOp(
    "/home/public/june29/2D/DMQ/Axial/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Axial_dataTest, DMQ_Axial_labelTest = get_inputOp(
    "/home/public/june29/2D/DMQ/Axial/" + i + "/Test.tfrecords",
    batch_size, batch_size)
DMQ_Coronal_dataTrain, DMQ_Coronal_labelTrain = get_inputOp(
    "/home/public/june29/2D/DMQ/Coronal/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Coronal_dataTest, DMQ_Coronal_labelTest = get_inputOp(
    "/home/public/june29/2D/DMQ/Coronal/" + i + "/Test.tfrecords",
    batch_size, batch_size)
DMQ_Sagittal_dataTrain, DMQ_Sagittal_labelTrain = get_inputOp(
    "/home/public/june29/2D/DMQ/Sagittal/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Sagittal_dataTest, DMQ_Sagittal_labelTest = get_inputOp(
    "/home/public/june29/2D/DMQ/Sagittal/" + i + "/Test.tfrecords",
    batch_size, batch_size)

sess = tf.InteractiveSession()
global_step = tf.Variable(0)
keep_prob = tf.placeholder(tf.float32)


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


def precision(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1
    precision = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(negative_position + 1) % 2])
    return precision


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


######################################   XOY_Frature   ###################################

x1 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
label1 = tf.placeholder(tf.float32, [None, 2])

inputData_1 = tf.reshape(x1, [-1, 28, 28, 1])

kernel_11 = weight_variable([5, 5, 1, 64])
bias_11 = bias_variable([64])
conv_11 = conv2d(inputData_1, kernel_11)
conv_out_11 = tf.nn.relu(conv_11 + bias_11)
pooling_out_11 = max_pool_2x2(conv_out_11)

kernel_12 = weight_variable([5, 5,64, 64])
bias_12 = bias_variable([64])
conv_12 = conv2d(pooling_out_11, kernel_12)
conv_out_12 = tf.nn.relu(conv_12 + bias_12)
pooling_out_12 = max_pool_2x2(conv_out_12)

pooling_out_12 = tf.reshape(pooling_out_12, [-1, 7 * 7 * 64])

w_fc_11 = weight_variable([7 * 7 * 64, 500])
b_fc_11 = bias_variable([500])
fc_out_11 = tf.nn.relu(tf.matmul(pooling_out_12, w_fc_11) + b_fc_11)
# drop11 = tf.nn.dropout(fc_out_11, keep_prob)

w_fc_12 = weight_variable([500, 250])
b_fc_12 = bias_variable([250])
fc_out_12 = tf.nn.relu(tf.matmul(fc_out_11, w_fc_12) + b_fc_12)

######################################   XOZ_Feature   ###################################

x2 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])

inputData_2 = tf.reshape(x2, [-1, 28, 28, 1])

kernel_21 = weight_variable([5, 5, 1, 64])
bias_21 = bias_variable([64])
conv_21 = conv2d(inputData_2, kernel_21)
conv_out_21 = tf.nn.relu(conv_21 + bias_21)
pooling_out_21 = max_pool_2x2(conv_out_21)

kernel_22 = weight_variable([5, 5, 64, 64])
bias_22 = bias_variable([64])
conv_22 = conv2d(pooling_out_21, kernel_22)
conv_out_22 = tf.nn.relu(conv_22 + bias_22)
pooling_out_22 = max_pool_2x2(conv_out_22)

pooling_out_22 = tf.reshape(pooling_out_22, [-1, 7 * 7 * 64])

w_fc_21 = weight_variable([7 * 7 * 64, 500])
b_fc_21 = bias_variable([500])
fc_out_21 = tf.nn.relu(tf.matmul(pooling_out_22, w_fc_21) + b_fc_21)
# drop21 = tf.nn.dropout(fc_out_21, keep_prob)

w_fc_22 = weight_variable([500, 250])
b_fc_22 = bias_variable([250])
fc_out_22 = tf.nn.relu(tf.matmul(fc_out_21, w_fc_22) + b_fc_22)

# ######################################   YOZ_Feature   ###################################
#
# x3 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
#
# inputData_3 = tf.reshape(x3, [-1, 28, 28, 1])
#
# kernel_31 = weight_variable([3,3, 1, 32])
# bias_31 = bias_variable([32])
# conv_31 = conv2d(inputData_3, kernel_31)
# conv_out_31 = tf.nn.relu(conv_31 + bias_31)
# pooling_out_31 = max_pool_2x2(conv_out_31)
#
# kernel_32 = weight_variable([3,5, 52, 64])
# bias_32 = bias_variable([64])
# conv_32 = conv2d(pooling_out_31, kernel_32)
# conv_out_32 = tf.nn.relu(conv_32 + bias_32)
# pooling_out_32 = max_pool_2x2(conv_out_32)
#
# pooling_out_32 = tf.reshape(pooling_out_32, [-1, 7 * 7 * 64])
#
# w_fc_31 = weight_variable([7 * 7 * 64, 500])
# b_fc_31 = bias_variable([500])
# fc_out_31 = tf.nn.relu(tf.matmul(pooling_out_32, w_fc_31) + b_fc_31)
# drop31 = tf.nn.dropout(fc_out_31, keep_prob)
#
# w_fc_32 = weight_variable([500, 250])
# b_fc_32 = bias_variable([250])
# fc_out_32 = tf.nn.relu(tf.matmul(drop31, w_fc_32) + b_fc_32)
#


###################   Common_Indevidual Analysis for XOY_Frature   ########################
common_V1 = weight_variable([250, 250])
bias_common1 = bias_variable([250])
common_feature1 = tf.nn.relu(tf.matmul(fc_out_12, common_V1) + bias_common1)

indevidal_Q1 = tf.placeholder(tf.float32, [250, 250])
bias_indevidual1 = bias_variable([250])
indevidual_feature1 = tf.nn.relu(tf.matmul(fc_out_12, indevidal_Q1) + bias_indevidual1)

###################   Common_Indevidual Analysis for XOZ_Frature   ########################
common_V2 = weight_variable([250, 250])
bias_common2 = bias_variable([250])
common_feature2 = tf.nn.relu(tf.matmul(fc_out_22, common_V2) + bias_common2)

indevidal_Q2 = tf.placeholder(tf.float32, [250, 250])
bias_indevidual2 = bias_variable([250])
indevidual_feature2 = tf.nn.relu(tf.matmul(fc_out_22, indevidal_Q2) + bias_indevidual2)

# ###################   Common_Indevidual Analysis for XOZ_Frature   ########################
# common_V3 = weight_variable([250, 250])
# bias_common3 = bias_variable([250])
# common_feature3 = tf.nn.relu(tf.matmul(fc_out_32, common_V3) + bias_common3)
#
# indevidal_Q3 = tf.placeholder(tf.float32, [250, 250])
# bias_indevidual3 = bias_variable([250])
# indevidual_feature3 = tf.nn.relu(tf.matmul(fc_out_32, indevidal_Q3) + bias_indevidual3)

######################################     Fusion      ####################################
# feature_cat = tf.concat(
#     [common_feature1, common_feature2, common_feature3, indevidual_feature1, indevidual_feature2, indevidual_feature3],
#     1)

feature_cat = tf.concat(
    [common_feature1, common_feature2, indevidual_feature1, indevidual_feature2],
    1)

# w_fc_f1 = weight_variable([250*6,500])
w_fc_f1 = weight_variable([1000, 250])
b_fc_f1 = bias_variable([250])
fc_f_out1 = tf.nn.relu(tf.matmul(feature_cat, w_fc_f1) + b_fc_f1)
drop1 = tf.nn.dropout(fc_f_out1, keep_prob)

w_fc_f2 = weight_variable([250, 2])
b_fc_f2 = bias_variable([2])
mid = tf.matmul(drop1, w_fc_f2) + b_fc_f2
prediction = tf.nn.softmax(mid)

with tf.name_scope('loss'):
    CL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))
    tf.summary.scalar('CL', CL)

    fai = 0.5 / 4125
    thta = 0.5 / 250

    part1_1 = tf.norm((common_feature1 - common_feature2), 'fro', axis=(0, 1))
    # part1_2 = tf.norm((common_feature1 - common_feature3), 'fro', axis=(0, 1))
    # part1_3 = tf.norm((common_feature2 - common_feature3), 'fro', axis=(0, 1))

    part2_1 = tf.norm((fc_out_12 - tf.matmul(fc_out_12, tf.matmul(tf.transpose(common_V1), common_V1)) - tf.matmul(
        fc_out_12, tf.matmul(tf.transpose(indevidal_Q1), indevidal_Q1))), 'fro', axis=(0, 1))
    part2_2 = tf.norm((fc_out_22 - tf.matmul(fc_out_22, tf.matmul(tf.transpose(common_V2), common_V2)) - tf.matmul(
        fc_out_22, tf.matmul(tf.transpose(indevidal_Q2), indevidal_Q2))), 'fro', axis=(0, 1))
    # part2_3 = tf.norm((fc_out_32 - tf.matmul(fc_out_32, tf.matmul(tf.transpose(common_V3), common_V3)) - tf.matmul(
    #     fc_out_32, tf.matmul(tf.transpose(indevidal_Q3), indevidal_Q3))), 'fro', axis=(0, 1))

    part3_1 = tf.norm((tf.matmul(tf.transpose(common_V1), indevidal_Q1)), 'fro', axis=(0, 1))
    part3_2 = tf.norm((tf.matmul(tf.transpose(common_V2), indevidal_Q2)), 'fro', axis=(0, 1))
    # part3_3 = tf.norm((tf.matmul(tf.transpose(common_V3), indevidal_Q3)), 'fro', axis=(0, 1))

    # loss_CI = (part1_1 + part1_2 + part1_3) + fai * (part2_1 + part2_2 + part2_3) + thta * (part3_1 + part3_2 + part3_3)
    loss_CI = (part1_1) + fai * (part2_1 + part2_2) + thta * (part3_1 + part3_2)

    total_loss = CL + loss_CI
    tf.summary.scalar('total_loss', total_loss)

with tf.name_scope('LR'):
    learning_rate = tf.train.exponential_decay(1e-5, global_step, decay_steps=4125 / 64, decay_rate=0.98,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    tf.summary.scalar('learning rate', learning_rate)

with tf.name_scope('Accuracy'):
    output_position = tf.argmax(prediction, 1)
    label_position = tf.argmax(label1, 1)
    predict = tf.equal(output_position, label_position)
    Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('Accuracy', Accuracy)

merge = tf.summary.merge_all()
#####################################    Train    ##########################################

sess.run(tf.global_variables_initializer())

board_path = '/home/public/june29/log/' + '2D_c+i'
if (not (os.path.exists(board_path))):
    os.mkdir(board_path)

test_board_path = board_path + '/' + 'test'
if (not (os.path.exists(test_board_path))):
    os.mkdir(test_board_path)
test_board_path = test_board_path + '/' + i
if (not (os.path.exists(test_board_path))):
    os.mkdir(test_board_path)

train_board_path = board_path + '/' + 'train'
if (not (os.path.exists(train_board_path))):
    os.mkdir(train_board_path)
train_board_path = train_board_path + '/' + i
if (not (os.path.exists(train_board_path))):
    os.mkdir(train_board_path)

test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())

tf.train.start_queue_runners(sess=sess)

before = time.time()

for times in range(10000):
    global_step = times

    common_V1_r = sess.run(common_V1)
    common_V2_r = sess.run(common_V2)
    # common_V3_r = sess.run(common_V3)

    indevidal_Q1_r = orth(common_V1_r)
    indevidal_Q2_r = orth(common_V2_r)
    # indevidal_Q3_r = orth(common_V3_r)

    if indevidal_Q1_r.shape != (250, 250):
        n1 = indevidal_Q1_r.shape[1]
        Q1 = np.zeros((250, 250))
        Q1[:, 0:n1] = indevidal_Q1_r
        indevidal_Q1_r = Q1

    if indevidal_Q2_r.shape != (250, 250):
        n2 = indevidal_Q2_r.shape[1]
        Q2 = np.zeros((250, 250))
        Q2[:, 0:n2] = indevidal_Q2_r
        indevidal_Q2_r = Q2

    # if indevidal_Q3_r.shape != (250, 250):
    #     n3 = indevidal_Q3_r.shape[1]
    #     Q3 = np.zeros((250, 250))
    #     Q3[:, 0:n3] = indevidal_Q3_r
    #     indevidal_Q3_r = Q3

    DMQ_Axial_dataTest_r, DMQ_Coronal_dataTest_r, DMQ_Sagittal_dataTest_r, \
    DMQ_Axial_labelTest_r, DMQ_Coronal_labelTest_r, DMQ_Sagittal_labelTest_r = sess.run(
        [DMQ_Axial_dataTest, DMQ_Coronal_dataTest, DMQ_Sagittal_dataTest, DMQ_Axial_labelTest,
         DMQ_Coronal_labelTest, DMQ_Sagittal_labelTest])
    DMQ_Axial_dataTrain_r, DMQ_Coronal_dataTrain_r, DMQ_Sagittal_dataTrain_r, \
    DMQ_Axial_labelTrain_r, DMQ_Coronal_labelTrain_r, DMQ_Sagittal_labelTrain_r = sess.run(
        [DMQ_Axial_dataTrain, DMQ_Coronal_dataTrain, DMQ_Sagittal_dataTrain,
         DMQ_Axial_labelTrain, DMQ_Coronal_labelTrain, DMQ_Sagittal_labelTrain])

    ###########################  test  #######################
    if times % 50 == 0:
        summary, acc, output_position_r, label_position_r, predict_r, p = sess.run(
            [merge, Accuracy, output_position, label_position, predict, prediction],
            # feed_dict={x1: DMQ_Axial_dataTest_r, x2: DMQ_Coronal_dataTest_r,
            #            x3: DMQ_Sagittal_dataTest_r, label1: DMQ_Axial_labelTest_r, indevidal_Q1: indevidal_Q1_r,
            #            indevidal_Q2: indevidal_Q2_r, indevidal_Q3: indevidal_Q3_r, keep_prob: 1.0})
            feed_dict={x1: DMQ_Axial_dataTest_r, x2: DMQ_Coronal_dataTest_r, label1: DMQ_Axial_labelTest_r,
                       indevidal_Q1: indevidal_Q1_r,
                       indevidal_Q2: indevidal_Q2_r, keep_prob: 1.0})

        sen, spe = Sensitivity_specificity(output_position_r, predict_r)
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        print('step : ' + str(times))
        print('Accuracy is: ' + str(acc) + '\n' + 'Sensitivity is: ' + str(
            sen) + '\n' + 'Specificity is: ' + str(spe))
        print('Auc : ' + str(roc_auc))
        print('\n')
        print(output_position_r)
        print(label_position_r)
        print('\n')
        # print(p, '\n')

    ###########################  show  #######################
    if times == 9999:
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        AUC = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic_2D_c+i')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % AUC)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    ###########################  train  #######################
    if times % 99 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merge, train_step],
                              # feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                              #            x3: DMQ_Sagittal_dataTrain_r,
                              #            label1: DMQ_Axial_labelTrain_r, indevidal_Q1: indevidal_Q1_r,
                              #            indevidal_Q2: indevidal_Q2_r, indevidal_Q3: indevidal_Q3_r, keep_prob: 0.5})
                              feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                         label1: DMQ_Axial_labelTrain_r, indevidal_Q1: indevidal_Q1_r,
                                         indevidal_Q2: indevidal_Q2_r, keep_prob: 0.5})
        train_writer.add_run_metadata(run_metadata, 'step%03d' % times)
        train_writer.add_summary(summary, times)
    else:
        # sess.run([train_step], feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
        #                                   x3: DMQ_Sagittal_dataTrain_r,
        #                                   label1: DMQ_Axial_labelTrain_r, indevidal_Q1: indevidal_Q1_r,
        #                                   indevidal_Q2: indevidal_Q2_r, indevidal_Q3: indevidal_Q3_r, keep_prob: 0.5})
        sess.run([train_step], feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                          label1: DMQ_Axial_labelTrain_r, indevidal_Q1: indevidal_Q1_r,
                                          indevidal_Q2: indevidal_Q2_r, keep_prob: 0.5})
after = time.time()
print('Total time is: ' + str((after - before) / 60) + ' minutes.')
train_writer.close()
test_writer.close()
