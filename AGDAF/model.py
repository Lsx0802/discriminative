import tensorflow as tf
import time
from sklearn.metrics import roc_curve, auc  # curve曲线
import os
from util import *

def Model(i, j, lr, ds, dr, t_step, batch_size, py_title, title, sess, AP_dataTrain,
                    AP_labelTrain, AP_dataTest, AP_labelTest,PVP_dataTrain, PVP_labelTrain,
                    PVP_dataTest, PVP_labelTest,PC_dataTrain, PC_labelTrain,PC_dataTest, PC_labelTest):
    global_step = tf.Variable(0)
    keep_prob = tf.placeholder(tf.float32)
    ######################################   AP_Frature   ###################################

    x1 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])
    label1 = tf.placeholder(tf.float32, [None, 2])

    inputData_1 = tf.reshape(x1, [-1, 16, 16, 16, 1])

    #
    W_conv11 = weight_variable([3, 3, 3, 1, 32])
    b_conv11 = bias_variable([32])
    h_conv11 = tf.nn.relu(conv3d(inputData_1, W_conv11) + b_conv11)

    W_q11 = weight_variable([1, 1, 1, 32, 16])
    h_q11 = conv3d(h_conv11, W_q11)
    h_q11 = tf.reshape(h_q11, [-1, 16 * 16 * 16, 16])

    W_k11 = weight_variable([1, 1, 1, 32, 16])
    h_k11 = conv3d(h_conv11, W_k11)
    h_k11 = tf.reshape(h_k11, [-1, 16 * 16 * 16, 16])
    h_k11 = tf.transpose(h_k11, [0, 2, 1])

    W_v11 = weight_variable([1, 1, 1, 32, 16])
    h_v11 = conv3d(h_conv11, W_v11)
    h_v11 = tf.reshape(h_v11, [-1, 16 * 16 * 16, 16])

    h_multi11 = tf.nn.softmax(tf.matmul(h_q11, h_k11))
    h_multi11 = tf.matmul(h_multi11, h_v11)

    h_out11 = tf.reshape(h_multi11, [-1, 16, 16, 16, 16])

    W_nl11 = weight_variable([1, 1, 1, 16, 32])
    h_nl11 = conv3d(h_out11, W_nl11)
    h_nl11 = tf.nn.relu(tf.add(h_nl11, h_conv11))

    ######################################   PVP_Feature   ###################################

    x2 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])
    label2 = tf.placeholder(tf.float32, [None, 2])

    inputData_2 = tf.reshape(x2, [-1, 16, 16, 16, 1])

    W_conv21 = weight_variable([3, 3, 3, 1, 32])
    b_conv21 = bias_variable([32])
    h_conv21 = tf.nn.relu(conv3d(inputData_2, W_conv21) + b_conv21)

    W_q21 = weight_variable([1, 1, 1, 32, 16])
    h_q21 = conv3d(h_conv21, W_q21)
    h_q21 = tf.reshape(h_q21, [-1, 16 * 16 * 16, 16])

    W_k21 = weight_variable([1, 1, 1, 32, 16])
    h_k21 = conv3d(h_conv21, W_k21)
    h_k21 = tf.reshape(h_k21, [-1, 16 * 16 * 16, 16])
    h_k21 = tf.transpose(h_k21, [0, 2, 1])

    W_v21 = weight_variable([1, 1, 1, 32, 16])
    h_v21 = conv3d(h_conv21, W_v21)
    h_v21 = tf.reshape(h_v21, [-1, 16 * 16 * 16, 16])

    h_multi21 = tf.nn.softmax(tf.matmul(h_q21, h_k21))
    h_multi21 = tf.matmul(h_multi21, h_v21)

    h_out21 = tf.reshape(h_multi21, [-1, 16, 16, 16, 16])

    W_nl21 = weight_variable([1, 1, 1, 16, 32])
    h_nl21 = conv3d(h_out21, W_nl21)
    h_nl21 = tf.nn.relu(tf.add(h_nl21, h_conv21))

    ######################################   PC_Feature   ###################################

    x3 = tf.placeholder(tf.float32, [None, 16 * 16 * 16 * 1])
    label3 = tf.placeholder(tf.float32, [None, 2])

    inputData_3 = tf.reshape(x3, [-1, 16, 16, 16, 1])

    W_conv31 = weight_variable([3, 3, 3, 1, 32])
    b_conv31 = bias_variable([32])
    h_conv31 = tf.nn.relu(conv3d(inputData_3, W_conv31) + b_conv31)

    W_q31 = weight_variable([1, 1, 1, 32, 16])
    h_q31 = conv3d(h_conv31, W_q31)
    h_q31 = tf.reshape(h_q31, [-1, 16 * 16 * 16, 16])

    W_k31 = weight_variable([1, 1, 1, 32, 16])
    h_k31 = conv3d(h_conv31, W_k31)
    h_k31 = tf.reshape(h_k31, [-1, 16 * 16 * 16, 16])
    h_k31 = tf.transpose(h_k31, [0, 2, 1])

    W_v31 = weight_variable([1, 1, 1, 32, 16])
    h_v31 = conv3d(h_conv31, W_v31)
    h_v31 = tf.reshape(h_v31, [-1, 16 * 16 * 16, 16])

    h_multi31 = tf.nn.softmax(tf.matmul(h_q31, h_k31))
    h_multi31 = tf.matmul(h_multi31, h_v31)

    h_out31 = tf.reshape(h_multi31, [-1, 16, 16, 16, 16])

    W_nl31 = weight_variable([1, 1, 1, 16, 32])
    h_nl31 = conv3d(h_out31, W_nl31)
    h_nl31 = tf.nn.relu(tf.add(h_nl31, h_conv31))

    ################ co 11
    W_q12 = weight_variable([1, 1, 1, 32, 16])
    h_q12 = conv3d(h_nl11, W_q12)
    h_q12 = tf.reshape(h_q12, [-1, 16 * 16 * 16, 16])

    W_k12 = weight_variable([1, 1, 1, 32, 16])
    h_k12 = conv3d(h_nl11, W_k12)
    h_k12 = tf.reshape(h_k12, [-1, 16 * 16 * 16, 16])
    h_k12 = tf.transpose(h_k12, [0, 2, 1])

    W_v12 = weight_variable([1, 1, 1, 32, 16])
    h_v12 = conv3d(h_nl11, W_v12)
    h_v12 = tf.reshape(h_v12, [-1, 16 * 16 * 16, 16])

    #####################co 21
    W_q22 = weight_variable([1, 1, 1, 32, 16])
    h_q22 = conv3d(h_nl21, W_q22)
    h_q22 = tf.reshape(h_q22, [-1, 16 * 16 * 16, 16])

    W_k22 = weight_variable([1, 1, 1, 32, 16])
    h_k22 = conv3d(h_nl21, W_k22)
    h_k22 = tf.reshape(h_k22, [-1, 16 * 16 * 16, 16])
    h_k22 = tf.transpose(h_k22, [0, 2, 1])

    W_v22 = weight_variable([1, 1, 1, 32, 16])
    h_v22 = conv3d(h_nl21, W_v22)
    h_v22 = tf.reshape(h_v22, [-1, 16 * 16 * 16, 16])

    ####################co 31
    W_q32 = weight_variable([1, 1, 1, 32, 16])
    h_q32 = conv3d(h_nl31, W_q32)
    h_q32 = tf.reshape(h_q32, [-1, 16 * 16 * 16, 16])

    W_k32 = weight_variable([1, 1, 1, 32, 16])
    h_k32 = conv3d(h_nl31, W_k32)
    h_k32 = tf.reshape(h_k32, [-1, 16 * 16 * 16, 16])
    h_k32 = tf.transpose(h_k32, [0, 2, 1])

    W_v32 = weight_variable([1, 1, 1, 32, 16])
    h_v32 = conv3d(h_nl31, W_v32)
    h_v32 = tf.reshape(h_v32, [-1, 16 * 16 * 16, 16])

    #################### mix 1

    h_multi12 = tf.nn.softmax(tf.matmul(h_q12, h_k22))
    h_multi12 = tf.matmul(h_multi12, h_v22)

    h_out12 = tf.reshape(h_multi12, [-1, 16, 16, 16, 16])

    W_nl12 = weight_variable([1, 1, 1, 16, 32])
    h_nl12 = conv3d(h_out12, W_nl12)
    h_nl12 = tf.nn.relu(tf.add(h_nl12, h_nl11))

    h_multi22 = tf.nn.softmax(tf.matmul(h_q22, h_k12))
    h_multi22 = tf.matmul(h_multi22, h_v12)

    h_out22 = tf.reshape(h_multi22, [-1, 16, 16, 16, 16])

    W_nl22 = weight_variable([1, 1, 1, 16, 32])
    h_nl22 = conv3d(h_out22, W_nl22)
    h_nl22 = tf.nn.relu(tf.add(h_nl22, h_nl21))

    h_multi32 = tf.nn.softmax(tf.matmul(h_q32, h_k12))
    h_multi32 = tf.matmul(h_multi32, h_v12)

    h_out32 = tf.reshape(h_multi32, [-1, 16, 16, 16, 16])

    W_nl32 = weight_variable([1, 1, 1, 16, 32])
    h_nl32 = conv3d(h_out32, W_nl32)
    h_nl32 = tf.nn.relu(tf.add(h_nl32, h_nl31))

    #################### mix 2

    h_multi122 = tf.nn.softmax(tf.matmul(h_q12, h_k32))
    h_multi122 = tf.matmul(h_multi122, h_v32)

    h_out122 = tf.reshape(h_multi122, [-1, 16, 16, 16, 16])

    W_nl122 = weight_variable([1, 1, 1, 16, 32])
    h_nl122 = conv3d(h_out122, W_nl122)
    h_nl122 = tf.nn.relu(tf.add(h_nl122, h_nl12))

    h_multi222 = tf.nn.softmax(tf.matmul(h_q22, h_k32))
    h_multi222 = tf.matmul(h_multi222, h_v32)

    h_out222 = tf.reshape(h_multi222, [-1, 16, 16, 16, 16])

    W_nl222 = weight_variable([1, 1, 1, 16, 32])
    h_nl222 = conv3d(h_out222, W_nl222)
    h_nl222 = tf.nn.relu(tf.add(h_nl222, h_nl22))

    h_multi322 = tf.nn.softmax(tf.matmul(h_q32, h_k22))
    h_multi322 = tf.matmul(h_multi322, h_v22)

    h_out322 = tf.reshape(h_multi322, [-1, 16, 16, 16, 16])

    W_nl322 = weight_variable([1, 1, 1, 16, 32])
    h_nl322 = conv3d(h_out322, W_nl322)
    h_nl322 = tf.nn.relu(tf.add(h_nl322, h_nl32))
    ##############
    h_pool11 = max_pool_2x2_3D(h_nl122)
    h_pool21 = max_pool_2x2_3D(h_nl222)
    h_pool31 = max_pool_2x2_3D(h_nl322)

    ######################################   AP_Frature 2  ###################################

    W_conv13 = weight_variable([3, 3, 3, 32, 64])
    b_conv13 = bias_variable([64])
    h_conv13 = tf.nn.relu(conv3d(h_pool11, W_conv13) + b_conv13)

    W_q13 = weight_variable([1, 1, 1, 64, 32])
    h_q13 = conv3d(h_conv13, W_q13)
    h_q13 = tf.reshape(h_q13, [-1, 8 * 8 * 8, 32])

    W_k13 = weight_variable([1, 1, 1, 64, 32])
    h_k13 = conv3d(h_conv13, W_k13)
    h_k13 = tf.reshape(h_k13, [-1, 8 * 8 * 8, 32])
    h_k13 = tf.transpose(h_k13, [0, 2, 1])

    W_v13 = weight_variable([1, 1, 1, 64, 32])
    h_v13 = conv3d(h_conv13, W_v13)
    h_v13 = tf.reshape(h_v13, [-1, 8 * 8 * 8, 32])

    h_multi13 = tf.nn.softmax(tf.matmul(h_q13, h_k13))
    h_multi13 = tf.matmul(h_multi13, h_v13)

    h_out13 = tf.reshape(h_multi13, [-1, 8, 8, 8, 32])

    W_nl13 = weight_variable([1, 1, 1, 32, 64])
    h_nl13 = conv3d(h_out13, W_nl13)
    h_nl13 = tf.nn.relu(tf.add(h_nl13, h_conv13))

    ######################################   PVP_Feature 2  ###################################

    W_conv23 = weight_variable([3, 3, 3, 32, 64])
    b_conv23 = bias_variable([64])
    h_conv23 = tf.nn.relu(conv3d(h_pool21, W_conv23) + b_conv23)

    W_q23 = weight_variable([1, 1, 1, 64, 32])
    h_q23 = conv3d(h_conv23, W_q23)
    h_q23 = tf.reshape(h_q23, [-1, 8 * 8 * 8, 32])

    W_k23 = weight_variable([1, 1, 1, 64, 32])
    h_k23 = conv3d(h_conv23, W_k23)
    h_k23 = tf.reshape(h_k23, [-1, 8 * 8 * 8, 32])
    h_k23 = tf.transpose(h_k23, [0, 2, 1])

    W_v23 = weight_variable([1, 1, 1, 64, 32])
    h_v23 = conv3d(h_conv23, W_v23)
    h_v23 = tf.reshape(h_v23, [-1, 8 * 8 * 8, 32])

    h_multi23 = tf.nn.softmax(tf.matmul(h_q23, h_k23))
    h_multi23 = tf.matmul(h_multi23, h_v23)

    h_out23 = tf.reshape(h_multi23, [-1, 8, 8, 8, 32])

    W_nl23 = weight_variable([1, 1, 1, 32, 64])
    h_nl23 = conv3d(h_out23, W_nl23)
    h_nl23 = tf.nn.relu(tf.add(h_nl23, h_conv23))

    ######################################   PC_Feature  2 ###################################

    W_conv33 = weight_variable([3, 3, 3, 32, 64])
    b_conv33 = bias_variable([64])
    h_conv33 = tf.nn.relu(conv3d(h_pool31, W_conv33) + b_conv33)

    W_q33 = weight_variable([1, 1, 1, 64, 32])
    h_q33 = conv3d(h_conv33, W_q33)
    h_q33 = tf.reshape(h_q33, [-1, 8 * 8 * 8, 32])

    W_k33 = weight_variable([1, 1, 1, 64, 32])
    h_k33 = conv3d(h_conv33, W_k33)
    h_k33 = tf.reshape(h_k33, [-1, 8 * 8 * 8, 32])
    h_k33 = tf.transpose(h_k33, [0, 2, 1])

    W_v33 = weight_variable([1, 1, 1, 64, 32])
    h_v33 = conv3d(h_conv33, W_v33)
    h_v33 = tf.reshape(h_v33, [-1, 8 * 8 * 8, 32])

    h_multi33 = tf.nn.softmax(tf.matmul(h_q33, h_k33))
    h_multi33 = tf.matmul(h_multi33, h_v33)

    h_out33 = tf.reshape(h_multi33, [-1, 8, 8, 8, 32])

    W_nl33 = weight_variable([1, 1, 1, 32, 64])
    h_nl33 = conv3d(h_out33, W_nl33)
    h_nl33 = tf.nn.relu(tf.add(h_nl33, h_conv33))

    ################ co 12
    W_q14 = weight_variable([1, 1, 1, 64, 32])
    h_q14 = conv3d(h_nl13, W_q14)
    h_q14 = tf.reshape(h_q14, [-1, 8 * 8 * 8, 32])

    W_k14 = weight_variable([1, 1, 1, 64, 32])
    h_k14 = conv3d(h_nl13, W_k14)
    h_k14 = tf.reshape(h_k14, [-1, 8 * 8 * 8, 32])
    h_k14 = tf.transpose(h_k14, [0, 2, 1])

    W_v14 = weight_variable([1, 1, 1, 64, 32])
    h_v14 = conv3d(h_nl13, W_v14)
    h_v14 = tf.reshape(h_v14, [-1, 8 * 8 * 8, 32])

    #####################co 22
    W_q24 = weight_variable([1, 1, 1, 64, 32])
    h_q24 = conv3d(h_nl23, W_q24)
    h_q24 = tf.reshape(h_q24, [-1, 8 * 8 * 8, 32])

    W_k24 = weight_variable([1, 1, 1, 64, 32])
    h_k24 = conv3d(h_nl23, W_k24)
    h_k24 = tf.reshape(h_k24, [-1, 8 * 8 * 8, 32])
    h_k24 = tf.transpose(h_k24, [0, 2, 1])

    W_v24 = weight_variable([1, 1, 1, 64, 32])
    h_v24 = conv3d(h_nl23, W_v24)
    h_v24 = tf.reshape(h_v24, [-1, 8 * 8 * 8, 32])

    ####################co 32
    W_q34 = weight_variable([1, 1, 1, 64, 32])
    h_q34 = conv3d(h_nl33, W_q34)
    h_q34 = tf.reshape(h_q34, [-1, 8 * 8 * 8, 32])

    W_k34 = weight_variable([1, 1, 1, 64, 32])
    h_k34 = conv3d(h_nl33, W_k34)
    h_k34 = tf.reshape(h_k34, [-1, 8 * 8 * 8, 32])
    h_k34 = tf.transpose(h_k34, [0, 2, 1])

    W_v34 = weight_variable([1, 1, 1, 64, 32])
    h_v34 = conv3d(h_nl33, W_v34)
    h_v34 = tf.reshape(h_v34, [-1, 8 * 8 * 8, 32])

    #################### mix 3

    h_multi14 = tf.nn.softmax(tf.matmul(h_q14, h_k24))
    h_multi14 = tf.matmul(h_multi14, h_v24)

    h_out14 = tf.reshape(h_multi14, [-1, 8, 8, 8, 32])

    W_nl14 = weight_variable([1, 1, 1, 32, 64])
    h_nl14 = conv3d(h_out14, W_nl14)
    h_nl14 = tf.nn.relu(tf.add(h_nl14, h_nl13))

    h_multi24 = tf.nn.softmax(tf.matmul(h_q24, h_k14))
    h_multi24 = tf.matmul(h_multi24, h_v14)

    h_out24 = tf.reshape(h_multi24, [-1, 8, 8, 8, 32])

    W_nl24 = weight_variable([1, 1, 1, 32, 64])
    h_nl24 = conv3d(h_out24, W_nl24)
    h_nl24 = tf.nn.relu(tf.add(h_nl24, h_nl23))

    h_multi34 = tf.nn.softmax(tf.matmul(h_q34, h_k14))
    h_multi34 = tf.matmul(h_multi34, h_v14)

    h_out34 = tf.reshape(h_multi34, [-1, 8, 8, 8, 32])

    W_nl34 = weight_variable([1, 1, 1, 32, 64])
    h_nl34 = conv3d(h_out34, W_nl34)
    h_nl34 = tf.nn.relu(tf.add(h_nl34, h_nl33))

    #################### mix 4

    h_multi142 = tf.nn.softmax(tf.matmul(h_q14, h_k34))
    h_multi142 = tf.matmul(h_multi142, h_v34)

    h_out142 = tf.reshape(h_multi142, [-1, 8, 8, 8, 32])

    W_nl142 = weight_variable([1, 1, 1, 32, 64])
    h_nl142 = conv3d(h_out142, W_nl142)
    h_nl142 = tf.nn.relu(tf.add(h_nl142, h_nl14))

    h_multi242 = tf.nn.softmax(tf.matmul(h_q24, h_k34))
    h_multi242 = tf.matmul(h_multi242, h_v34)

    h_out242 = tf.reshape(h_multi242, [-1, 8, 8, 8, 32])

    W_nl242 = weight_variable([1, 1, 1, 32, 64])
    h_nl242 = conv3d(h_out242, W_nl242)
    h_nl242 = tf.nn.relu(tf.add(h_nl242, h_nl24))

    h_multi342 = tf.nn.softmax(tf.matmul(h_q34, h_k24))
    h_multi342 = tf.matmul(h_multi342, h_v24)

    h_out342 = tf.reshape(h_multi342, [-1, 8, 8, 8, 32])

    W_nl342 = weight_variable([1, 1, 1, 32, 64])
    h_nl342 = conv3d(h_out342, W_nl342)
    h_nl342 = tf.nn.relu(tf.add(h_nl342, h_nl34))
    ##############
    h_pool12 = max_pool_2x2_3D(h_nl142)
    h_pool22 = max_pool_2x2_3D(h_nl242)
    h_pool32 = max_pool_2x2_3D(h_nl342)

    ######################################   AP_Frature 3  ###################################

    #
    W_conv15 = weight_variable([3, 3, 3, 64, 64])
    b_conv15 = bias_variable([64])
    h_conv15 = tf.nn.relu(conv3d(h_pool12, W_conv15) + b_conv15)

    W_q15 = weight_variable([1, 1, 1, 64, 32])
    h_q15 = conv3d(h_conv15, W_q15)
    h_q15 = tf.reshape(h_q15, [-1, 4 * 4 * 4, 32])

    W_k15 = weight_variable([1, 1, 1, 64, 32])
    h_k15 = conv3d(h_conv15, W_k15)
    h_k15 = tf.reshape(h_k15, [-1, 4 * 4 * 4, 32])
    h_k15 = tf.transpose(h_k15, [0, 2, 1])

    W_v15 = weight_variable([1, 1, 1, 64, 32])
    h_v15 = conv3d(h_conv15, W_v15)
    h_v15 = tf.reshape(h_v15, [-1, 4 * 4 * 4, 32])

    h_multi15 = tf.nn.softmax(tf.matmul(h_q15, h_k15))
    h_multi15 = tf.matmul(h_multi15, h_v15)

    h_out15 = tf.reshape(h_multi15, [-1, 4, 4, 4, 32])

    W_nl15 = weight_variable([1, 1, 1, 32, 64])
    h_nl15 = conv3d(h_out15, W_nl15)
    h_nl15 = tf.nn.relu(tf.add(h_nl15, h_conv15))

    ######################################   PVP_Feature 3  ###################################

    W_conv25 = weight_variable([3, 3, 3, 64, 64])
    b_conv25 = bias_variable([64])
    h_conv25 = tf.nn.relu(conv3d(h_pool22, W_conv25) + b_conv25)

    W_q25 = weight_variable([1, 1, 1, 64, 32])
    h_q25 = conv3d(h_conv25, W_q25)
    h_q25 = tf.reshape(h_q25, [-1, 4 * 4 * 4, 32])

    W_k25 = weight_variable([1, 1, 1, 64, 32])
    h_k25 = conv3d(h_conv25, W_k25)
    h_k25 = tf.reshape(h_k25, [-1, 4 * 4 * 4, 32])
    h_k25 = tf.transpose(h_k25, [0, 2, 1])

    W_v25 = weight_variable([1, 1, 1, 64, 32])
    h_v25 = conv3d(h_conv25, W_v25)
    h_v25 = tf.reshape(h_v25, [-1, 4 * 4 * 4, 32])

    h_multi25 = tf.nn.softmax(tf.matmul(h_q25, h_k25))
    h_multi25 = tf.matmul(h_multi25, h_v25)

    h_out25 = tf.reshape(h_multi25, [-1, 4, 4, 4, 32])

    W_nl25 = weight_variable([1, 1, 1, 32, 64])
    h_nl25 = conv3d(h_out25, W_nl25)
    h_nl25 = tf.nn.relu(tf.add(h_nl25, h_conv25))

    ######################################   PC_Feature 3  ###################################

    W_conv35 = weight_variable([3, 3, 3, 64, 64])
    b_conv35 = bias_variable([64])
    h_conv35 = tf.nn.relu(conv3d(h_pool32, W_conv35) + b_conv35)

    W_q35 = weight_variable([1, 1, 1, 64, 32])
    h_q35 = conv3d(h_conv35, W_q35)
    h_q35 = tf.reshape(h_q35, [-1, 4 * 4 * 4, 32])

    W_k35 = weight_variable([1, 1, 1, 64, 32])
    h_k35 = conv3d(h_conv35, W_k35)
    h_k35 = tf.reshape(h_k35, [-1, 4 * 4 * 4, 32])
    h_k35 = tf.transpose(h_k35, [0, 2, 1])

    W_v35 = weight_variable([1, 1, 1, 64, 32])
    h_v35 = conv3d(h_conv35, W_v35)
    h_v35 = tf.reshape(h_v35, [-1, 4 * 4 * 4, 32])

    h_multi35 = tf.nn.softmax(tf.matmul(h_q35, h_k35))
    h_multi35 = tf.matmul(h_multi35, h_v35)

    h_out35 = tf.reshape(h_multi35, [-1, 4, 4, 4, 32])

    W_nl35 = weight_variable([1, 1, 1, 32, 64])
    h_nl35 = conv3d(h_out35, W_nl35)
    h_nl35 = tf.nn.relu(tf.add(h_nl35, h_conv35))

    ################ co axial
    W_q16 = weight_variable([1, 1, 1, 64, 32])
    h_q16 = conv3d(h_nl15, W_q16)
    h_q16 = tf.reshape(h_q16, [-1, 4 * 4 * 4, 32])

    W_k16 = weight_variable([1, 1, 1, 64, 32])
    h_k16 = conv3d(h_nl15, W_k16)
    h_k16 = tf.reshape(h_k16, [-1, 4 * 4 * 4, 32])
    h_k16 = tf.transpose(h_k16, [0, 2, 1])

    W_v16 = weight_variable([1, 1, 1, 64, 32])
    h_v16 = conv3d(h_nl15, W_v16)
    h_v16 = tf.reshape(h_v16, [-1, 4 * 4 * 4, 32])

    #####################co con
    W_q26 = weight_variable([1, 1, 1, 64, 32])
    h_q26 = conv3d(h_nl25, W_q26)
    h_q26 = tf.reshape(h_q26, [-1, 4 * 4 * 4, 32])

    W_k26 = weight_variable([1, 1, 1, 64, 32])
    h_k26 = conv3d(h_nl25, W_k26)
    h_k26 = tf.reshape(h_k26, [-1, 4 * 4 * 4, 32])
    h_k26 = tf.transpose(h_k26, [0, 2, 1])

    W_v26 = weight_variable([1, 1, 1, 64, 32])
    h_v26 = conv3d(h_nl25, W_v26)
    h_v26 = tf.reshape(h_v26, [-1, 4 * 4 * 4, 32])

    ####################co s
    W_q36 = weight_variable([1, 1, 1, 64, 32])
    h_q36 = conv3d(h_nl35, W_q36)
    h_q36 = tf.reshape(h_q36, [-1, 4 * 4 * 4, 32])

    W_k36 = weight_variable([1, 1, 1, 64, 32])
    h_k36 = conv3d(h_nl35, W_k36)
    h_k36 = tf.reshape(h_k36, [-1, 4 * 4 * 4, 32])
    h_k36 = tf.transpose(h_k36, [0, 2, 1])

    W_v36 = weight_variable([1, 1, 1, 64, 32])
    h_v36 = conv3d(h_nl35, W_v36)
    h_v36 = tf.reshape(h_v36, [-1, 4 * 4 * 4, 32])

    #################### mix 5

    h_multi16 = tf.nn.softmax(tf.matmul(h_q16, h_k26))
    h_multi16 = tf.matmul(h_multi16, h_v26)

    h_out16 = tf.reshape(h_multi16, [-1, 4, 4, 4, 32])

    W_nl16 = weight_variable([1, 1, 1, 32, 64])
    h_nl16 = conv3d(h_out16, W_nl16)
    h_nl16 = tf.nn.relu(tf.add(h_nl16, h_nl15))

    h_multi26 = tf.nn.softmax(tf.matmul(h_q26, h_k16))
    h_multi26 = tf.matmul(h_multi26, h_v16)

    h_out26 = tf.reshape(h_multi26, [-1, 4, 4, 4, 32])

    W_nl26 = weight_variable([1, 1, 1, 32, 64])
    h_nl26 = conv3d(h_out26, W_nl26)
    h_nl26 = tf.nn.relu(tf.add(h_nl26, h_nl25))

    h_multi36 = tf.nn.softmax(tf.matmul(h_q36, h_k16))
    h_multi36 = tf.matmul(h_multi36, h_v16)

    h_out36 = tf.reshape(h_multi36, [-1, 4, 4, 4, 32])

    W_nl36 = weight_variable([1, 1, 1, 32, 64])
    h_nl36 = conv3d(h_out36, W_nl36)
    h_nl36 = tf.nn.relu(tf.add(h_nl36, h_nl35))

    #################### mix 6

    h_multi162 = tf.nn.softmax(tf.matmul(h_q16, h_k36))
    h_multi162 = tf.matmul(h_multi162, h_v36)

    h_out162 = tf.reshape(h_multi162, [-1, 4, 4, 4, 32])

    W_nl162 = weight_variable([1, 1, 1, 32, 64])
    h_nl162 = conv3d(h_out162, W_nl162)
    h_nl162 = tf.nn.relu(tf.add(h_nl162, h_nl16))

    h_multi262 = tf.nn.softmax(tf.matmul(h_q26, h_k36))
    h_multi262 = tf.matmul(h_multi262, h_v36)

    h_out262 = tf.reshape(h_multi262, [-1, 4, 4, 4, 32])

    W_nl262 = weight_variable([1, 1, 1, 32, 64])
    h_nl262 = conv3d(h_out262, W_nl262)
    h_nl262 = tf.nn.relu(tf.add(h_nl262, h_nl26))

    h_multi362 = tf.nn.softmax(tf.matmul(h_q36, h_k26))
    h_multi362 = tf.matmul(h_multi362, h_v26)

    h_out362 = tf.reshape(h_multi362, [-1, 4, 4, 4, 32])

    W_nl362 = weight_variable([1, 1, 1, 32, 64])
    h_nl362 = conv3d(h_out362, W_nl362)
    h_nl362 = tf.nn.relu(tf.add(h_nl362, h_nl36))
    ##############
    h_pool13 = max_pool_2x2_3D(h_nl162)
    h_pool23 = max_pool_2x2_3D(h_nl262)
    h_pool33 = max_pool_2x2_3D(h_nl362)

    ###########

    W_fc_nl11 = weight_variable([2 * 2 * 2 * 64, 500])
    b_fc_nl11 = bias_variable([500])
    h_pool_flat_nl11 = tf.reshape(h_pool13, [-1, 2 * 2 * 2 * 64])
    h_fc_nl11 = tf.nn.relu(tf.matmul(h_pool_flat_nl11, W_fc_nl11) + b_fc_nl11)
    drop11 = tf.nn.dropout(h_fc_nl11, keep_prob)

    W_fc_nl12 = weight_variable([500, 50])
    b_fc_nl12 = bias_variable([50])
    h_fc_nl12 = tf.nn.relu(tf.matmul(drop11, W_fc_nl12) + b_fc_nl12)

    W_1 = W_variable([50, 50])
    W_1_F = tf.matmul(h_fc_nl12, W_1)

    w_fc_13 = weight_variable([50, 2])
    b_fc_13 = bias_variable([2])
    mid1 = tf.matmul(W_1_F, w_fc_13) + b_fc_13

    with tf.name_scope('Loss_AP'):
        L1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid1, labels=label1))
        tf.summary.scalar('Loss_AP', L1)

    ####################
    W_fc_nl21 = weight_variable([2 * 2 * 2 * 64, 500])
    b_fc_nl21 = bias_variable([500])
    h_pool_flat_nl21 = tf.reshape(h_pool23, [-1, 2 * 2 * 2 * 64])
    h_fc_nl21 = tf.nn.relu(tf.matmul(h_pool_flat_nl21, W_fc_nl21) + b_fc_nl21)
    drop21 = tf.nn.dropout(h_fc_nl21, keep_prob)

    W_fc_nl22 = weight_variable([500, 50])
    b_fc_nl22 = bias_variable([50])
    h_fc_nl22 = tf.nn.relu(tf.matmul(drop21, W_fc_nl22) + b_fc_nl22)

    W_2 = W_variable([50, 50])
    W_2_F = tf.matmul(h_fc_nl22, W_2)

    w_fc_23 = weight_variable([50, 2])
    b_fc_23 = bias_variable([2])
    mid2 = tf.matmul(W_2_F, w_fc_23) + b_fc_23

    with tf.name_scope('Loss_PVP'):
        L2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid2, labels=label1))
        tf.summary.scalar('Loss_PVP', L2)

    #######################
    W_fc_nl31 = weight_variable([2 * 2 * 2 * 64, 500])
    b_fc_nl31 = bias_variable([500])
    h_pool_flat_nl31 = tf.reshape(h_pool33, [-1, 2 * 2 * 2 * 64])
    h_fc_nl31 = tf.nn.relu(tf.matmul(h_pool_flat_nl31, W_fc_nl31) + b_fc_nl31)
    drop31 = tf.nn.dropout(h_fc_nl31, keep_prob)

    W_fc_nl32 = weight_variable([500, 50])
    b_fc_nl32 = bias_variable([50])
    h_fc_nl32 = tf.nn.relu(tf.matmul(drop31, W_fc_nl32) + b_fc_nl32)

    W_3 = W_variable([50, 50])
    W_3_F = tf.matmul(h_fc_nl32, W_3)

    w_fc_33 = weight_variable([50, 2])
    b_fc_33 = bias_variable([2])
    mid3 = tf.matmul(W_3_F, w_fc_33) + b_fc_33

    with tf.name_scope('Loss_PC'):
        L3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid3, labels=label1))
        tf.summary.scalar('Loss_PC', L3)

    ##########################################################################
    feature_cat = tf.concat([W_1_F, W_2_F, W_3_F], 1)

    w_fc_f1 = weight_variable([150, 50])
    b_fc_f1 = bias_variable([50])
    fc_out_f1 = tf.matmul(feature_cat, w_fc_f1) + b_fc_f1

    w_fc_f2 = weight_variable([50, 2])
    b_fc_f2 = bias_variable([2])
    mid = tf.matmul(fc_out_f1, w_fc_f2) + b_fc_f2

    prediction = tf.nn.softmax(mid)

    ################################################
    with tf.name_scope('loss'):

        loss_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))
        tf.summary.scalar('loss_con', loss_cross_entropy)

        ###################################
        miu = 100.0
        tao = 1.0

        D_AP = pairwise_distance(W_1_F, label1, miu, tao, batch_size)
        D_PVP = pairwise_distance(W_2_F, label2, miu, tao, batch_size)
        D_PC = pairwise_distance(W_3_F, label3, miu, tao, batch_size)

        loss_D = D_AP + D_PVP + D_PC
        tf.summary.scalar('loss_D', loss_D)

        ##################################
        beta = 2.0

        a1 = S(L1, L2, beta) + S(L1, L3, beta)
        a2 = S(L2, L1, beta) + S(L2, L3, beta)
        a3 = S(L3, L1, beta) + S(L3, L2, beta)

        ################################
        total_loss = loss_cross_entropy + a1 * L1 + a2 * L2 + a3 * L3 + loss_D
        tf.summary.scalar('total_loss', total_loss)

    learning_rate = tf.train.exponential_decay(lr, global_step, decay_stePC=ds, decay_rate=dr,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    with tf.name_scope('Accuracy'):
        output_position = tf.argmax(prediction, 1)
        label_position = tf.argmax(label1, 1)
        predict = tf.equal(output_position, label_position)
        Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
        tf.summary.scalar('Accuracy', Accuracy)

    merge = tf.summary.merge_all()
    #####################################    Train    ##########################################

    sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()

    board_path = "/home/public/AGDAF/log/" + py_title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + i
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + j
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    test_board_path = board_path + '/' + 'test'
    if (not (os.path.exists(test_board_path))):
        os.mkdir(test_board_path)

    train_board_path = board_path + '/' + 'train'
    if (not (os.path.exists(train_board_path))):
        os.mkdir(train_board_path)

    test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
    train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)

    before = time.time()
    trigger=1024*1024
    flag=0
    early_stopping=100

    for times in range(t_step):
        AP_dataTest_r, AP_labelTest_r, PVP_dataTest_r, PVP_labelTest_r, \
        PC_dataTest_r, PC_labelTest_r, = sess.run(
            [AP_dataTest, AP_labelTest, PVP_dataTest, PVP_labelTest, PC_dataTest, PC_labelTest])
        AP_dataTrain_r, AP_labelTrain_r, PVP_dataTrain_r, PVP_labelTrain_r, \
        PC_dataTrain_r, PC_labelTrain_r = sess.run(
            [AP_dataTrain, AP_labelTrain, PVP_dataTrain, PVP_labelTrain, PC_dataTrain,
             PC_labelTrain])

        ###########################  test  #######################
        if times % 10 == 0:
            total_loss_r, summary, acc, output_position_r, label_position_r, \
            predict_r, p, loss_cross_entropy_r = sess.run(
                [total_loss, merge, Accuracy, output_position,
                 label_position, predict, prediction, loss_cross_entropy],
                feed_dict={x1: AP_dataTest_r, label1: AP_labelTest_r, x2: PVP_dataTest_r,
                           label2: PVP_labelTest_r, x3: PC_dataTest_r, label3: PC_labelTest_r,
                           keep_prob: 1.0})
            test_writer.add_summary(summary, times)
            sen, spe = Sensitivity_specificity(output_position_r, predict_r)
            fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
            AUC = auc(fpr, tpr)

            if trigger<total_loss_r:
                trigger=total_loss_r
                flag=0

            if flag==early_stopping:
                break
        ###########################  train  #######################
        if times % 99 == 0:
            summary, _ = sess.run([merge, train_step],
                                  feed_dict={x1: AP_dataTrain_r, label1: AP_labelTrain_r,
                                             x2: PVP_dataTrain_r, label2: PVP_labelTrain_r,
                                             x3: PC_dataTrain_r, label3: PC_labelTrain_r,
                                             keep_prob: 0.5})
            train_writer.add_summary(summary, times)
        else:
            sess.run([train_step],
                     feed_dict={x1: AP_dataTrain_r, label1: AP_labelTrain_r, x2: PVP_dataTrain_r,
                                label2: PVP_labelTrain_r, x3: PC_dataTrain_r,
                                label3: PC_labelTrain_r, keep_prob: 0.5})


    after = time.time()
    train_writer.close()
    test_writer.close()

    return acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before