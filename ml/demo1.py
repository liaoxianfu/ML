from __future__ import print_function
import tensorflow as tf


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
node3 = tf.add(node2, node1)
sess = tf.Session()
a = sess.run([node1, node2])
print(node1, node2)
print(a)

print(node3)
print(sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b


print(sess.run(adder_node*3, {a: 3, b: 4.5}))
print(sess.run(adder_node*3, {a: [4, 5, 6], b: [7, 8, 9, ]}))
