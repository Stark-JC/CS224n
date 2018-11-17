import tensorflow as tf


def cal(x, h, scope, time):
    with tf.variable_scope(scope):
        if time > 0:
            tf.get_variable_scope().reuse_variables()
        wx = tf.get_variable("w_x", shape=(1,), initializer=tf.random_uniform_initializer)
        return wx * x + h


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32)
    with tf.Session() as sess:
        with tf.variable_scope("step"):
            h = tf.constant(0, dtype=tf.float32)
            for i in range(3):
                h = cal(x, h, scope="step", time=i)
            sess.run(tf.global_variables_initializer())
            h = sess.run(h, feed_dict={x: 1})
            writer = tf.summary.FileWriter("./", sess.graph)
writer.close()
