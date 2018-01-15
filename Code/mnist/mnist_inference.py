import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    W = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    if (regularizer != None):
        tf.add_to_collection("losses", regularizer(W))
    return W

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        W = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        b = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, W) + b)
            
    with tf.variable_scope('layer2'):
        W = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        b = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, W) + b
            
    return layer2