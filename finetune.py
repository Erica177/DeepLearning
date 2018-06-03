"""
writen by erica shuai
"""

import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.contrib.data import Iterator

learning_rate = 0.001
num_epochs = 50  # 迭代次数
batch_size = 128
dropout_rate = 0.8
num_classes = 2  # 类别标签
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 5

filewriter_path = "F:/class/DeepLearning/work/tensorboard"  # 存储tensorboard文件
checkpoint_path = "F:/class/DeepLearning/work/checkpoints"  # 训练好的模型和参数存放目录

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

train_image_path = 'F:/class/DeepLearning/work/train/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）
test_image_cat_path = 'F:/class/DeepLearning/work/test/cat/'  # 指定测试集数据路径（根据实际情况指定测试数据集的路径）
test_image_dog_path = 'F:/class/DeepLearning/work/test/dog/'  # 指定测试集数据路径（根据实际情况指定测试数据集的路径）

label_path = []#cat_0 dog_1
test_label = []

# 打开训练数据集目录，读取全部图片，生成图片路径列表
image_path = np.array(glob.glob(train_image_path + 'cat.*.jpg')).tolist()
image_path_dog = np.array(glob.glob(train_image_path + 'dog.*.jpg')).tolist()
image_path[len(image_path):len(image_path)] = image_path_dog
for i in range(len(image_path)):
    if 'dog' in image_path[i]:
        label_path.append(1)
    else:
        label_path.append(0)

# 打开测试数据集目录，读取全部图片，生成图片路径列表
test_image = np.array(glob.glob(test_image_cat_path + '*.jpg')).tolist()
test_image_path_dog = np.array(glob.glob(test_image_dog_path + '*.jpg')).tolist()
test_image[len(test_image):len(test_image)] = test_image_path_dog

for i in range(len(test_image)):
    if 'dog' in test_image[i]:
        test_label.append(1)
    else:
        test_label.append(0)

with tf.name_scope('input'):
  # 调用图片生成器，把训练集图片转换成三维数组
    tr_data = ImageDataGenerator(
    images=image_path,
    mode='training',
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)
# 调用图片生成器，把测试集图片转换成三维数组
    test_data = ImageDataGenerator(
    images=test_image,
    mode='inference',
    labels=test_label,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False)
    # 定义迭代器
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)

    # 定义每次迭代的数据
    next_batch = iterator.get_next()

training_initalize=iterator.make_initializer(tr_data.data)
testing_initalize=iterator.make_initializer(test_data.data)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 图片数据通过AlexNet网络处理
model = AlexNet(x, keep_prob, num_classes, train_layers,'DEFAULT')
#执行网络图
score = model.fc8
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#print(var_list)

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)
    
tf.summary.scalar('cross_entropy', loss)
# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 把精确度加入到Tensorboard
tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge([tf.summary.scalar('cross_entropy', loss),
                                   tf.summary.scalar('accuracy', accuracy)])

writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

# 定义一代的迭代次数A
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) as sess:
    sess.run(tf.global_variables_initializer())

    # 把模型图加入Tensorboard
    writer.add_graph(sess.graph)

    # 把训练好的权重加入未训练的网络中
    #model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # 总共训练10代
    for epoch in range(num_epochs):
        sess.run(training_initalize)
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))

        #开始训练每一代
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            #if step % 15 == 0:
              #learning_rate *= 10
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob:1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)
           
        # 测试模型精确度
        print("{} Start validation".format(datetime.now()))
        sess.run(testing_initalize)
        test_acc = 0.
        test_count = 0

        for step in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.0})
            test_acc += acc
            test_count += 1
        test_acc /= test_count

        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # 把训练好的模型存储起来
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))
