import tensorflow as tf

# 创建变量
a = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([0.3], dtype=tf.float32)
# 创建因变量
x = tf.placeholder(dtype=tf.float32)
# 创建模型
linear_model = a * x + b

# 创建预测值
y = tf.placeholder(dtype=tf.float32)
# 创建损失函数
loss = tf.reduce_sum(tf.square(linear_model-y))

# 创建最优控制器
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 训练
train = optimizer.minimize(loss)

# 创建训练数据集
x_train = [1, 2, 3, 4, 5]
y_train = [5, 4, 3, 2, 1]

# 全局初始化
init = tf.global_variables_initializer()
# 创建会话层
sess = tf.Session()
sess.run(init)
# 开始迭代训练
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
# 结束迭代

# 迭代值获取
cur_a, cur_b, cur_loss = sess.run([a, b, loss], {x: x_train, y:y_train})

print(cur_loss,cur_a,cur_b)


