
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
mnist = input_data.read_data_sets("./MNIST DATA", one_hot=True)


# In[3]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAING_STEP = 3000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./mnist/save"
MODEL_NAME = "model.ckpt"


# In[4]:


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name="X-input")
    y = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name="Y-input")
    
    regulatizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    y_ = mnist_inference.inference(x, regulatizer)
    
    #定義存儲訓練輪數的變量
    global_step = tf.Variable(0, trainable=False)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)   
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_ , 
                                                                   labels=tf.argmax(y, 1))
    #計算當前batch中所有cross_entropy平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                               global_step, 
                                               mnist.train.num_examples / BATCH_SIZE, 
                                               LEARNING_RATE_DECAY)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    #先不執行optimizer先執行variable_averages_op
    with tf.control_dependencies([optimizer, variable_averages_op]):
        optimizer = tf.no_op(name="train")
    
    #初始化持久化類
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(6000):
                              
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            _, loss_value, step = sess.run([optimizer, loss, global_step], feed_dict={x : xs, y:ys})
            
            if (i % 1000 == 0):
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        
def main(argv=None):
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


# In[ ]:


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.app.run()


# In[ ]:




