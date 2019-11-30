from main import get_files, get_batch, inference, losses, trainning, evaluation
import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
N_CLASSES = 2 # 二分类问题，只有是还是否，即0，1
IMG_W = 208 #图片的宽度
IMG_H = 208 #图片的高度
BATCH_SIZE = 16 #批次大小
CAPACITY = 2000  # 队列最大容量2000
MAX_STEP = 2000 #最大训练步骤
learning_rate = 0.0001  #学习率

"""    
##1.数据的处理    
"""
# 训练图片路径
train_dir = 'train/'
# 输出log的位置
logs_train_dir = 'model/'
# 模型输出
train_model_dir = 'model/'
# 获取数据中的训练图片 和 训练标签
train, train_label = get_files()
# 获取转换的TensorFlow 张量
train_batch, train_label_batch = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)


"""    
##2.网络的推理    
"""
# 进行前向训练，获得回归值
train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)

"""    
##3.定义交叉熵和要使用的梯度下降的优化器     
"""
# 计算获得损失值loss
train_loss = losses(train_logits, train_label_batch)
# 对损失值进行优化
train_op = trainning(train_loss, learning_rate)

"""    
##4.定义后面要使用的变量    
"""
# 根据计算得到的损失值，计算出分类准确率
train__acc = evaluation(train_logits, train_label_batch)
# 将图形、训练过程合并在一起
summary_op = tf.summary.merge_all()


# 新建会话
sess = tf.Session()


# 将训练日志写入到logs_train_dir的文件夹内
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()  # 保存变量

# 执行训练过程，初始化变量
sess.run(tf.global_variables_initializer())


# 创建一个线程协调器，用来管理之后在Session中启动的所有线程
coord = tf.train.Coordinator()
# 启动入队的线程，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

"""    
进行训练：    
使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，    
会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;    
"""

try:
    for step in np.arange(MAX_STEP): #从0 到 2000 次 循环
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

        # 每50步打印一次损失值和准确率
        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))

            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)


        # 每2000步保存一次训练得到的模型
        if step % 500 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

# 如果读取到文件队列末尾会抛出此异常
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    # 使用coord.request_stop()来发出终止所有线程的命令

coord.join(threads)            # coord.join(threads)把线程加入主线程，等待threads结束
sess.close()                   # 关闭会话
