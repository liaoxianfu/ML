import tensorflow as tf

# 创建常量
const_num = tf.constant(0.5, dtype=tf.float32)

# 创建变量
a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)


# 创建占位符
x = tf.placeholder(tf.float32)

liner_model = a*x+b

# 注意，常量在调用时初始化。数值不会再次改变
# 变量在被调用时不会被初始化tf.Variable需要自己手动初始化
# 显示初始化
# 创建session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(liner_model, {x: [1, 2, 3, 4, 5]}))

# 创建损失函数
y = tf.placeholder(dtype=tf.float32)
square_deltas = tf.square(liner_model-y)
loss = tf.reduce_sum(square_deltas)
print(sess.run(loss, {x: [1, 2, 3,  4, ], y: [0, 0.3, 0.6, 0.9]}))

# 重新分配a,b的数值，完美初始化
fix_a = tf.assign(a, [-1])
fix_b = tf.assign(b, [1])
sess.run([fix_a, fix_b])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1, 2, 3, 4], y:[5, 6, 7, 8]})

print(sess.run([a,b]))














