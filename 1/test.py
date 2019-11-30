import tensorflow.compat.v1 as tf
from main import inference
import numpy as np
import pandas as pd
import math
from PIL import Image
import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tempfile

def get_one_image_file(img_dir):
    image = Image.open(img_dir)
    image = image.resize([208, 208])
    image = np.array(image)

    return image

def get_batch(image, image_w, image_h, batch_size):
    image = tf.cast(image, tf.string)
    image_contents = tf.read_file(image[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, image_w, image_h, 16])

    image_batch = tf.cast(image, tf.float32)
    print('batch 启动')
    return image_batch

def evaluate_one_image(image_array):
    # 数据集路径

    with tf.Graph().as_default():
        BATCH_SIZE = 1   # 获取一张图片
        N_CLASSES = 2  #二分类
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])     #inference输入数据需要是4维数据，需要对image进行resize

        logit = inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)    #inference的softmax层没有激活函数，这里增加激活函数

        #因为只有一副图，数据量小，所以用placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        #
        # 训练模型路径
        logs_train_dir = './model/'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 从指定路径下载模型
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 得到概率最大的索引
            max_index = np.argmax(prediction)
            return max_index
            # if max_index==0:
            #     print('This is a cat with possibility %.6f' %prediction[:, 0])
            # else:
            #     print('This is a dog with possibility %.6f' %prediction[:, 1])



def main():
    test_images = []
    for file in os.listdir('test/'):
        name = file.split('.')
        test_images.append('test/'+file)
    print('there are %d test images' % (len(test_images)))

    image_list = test_images
    print(image_list[1])

    prediction = []
    for il in image_list:
        image_array=get_one_image_file(il)
        prediction.append(evaluate_one_image(image_array))

    output = pd.DataFrame(prediction)
    output.to_csv("result.csv", index = True, header = False)
if __name__ == '__main__':
    main()
