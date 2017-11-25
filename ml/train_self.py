import tensorflow as tf

# 创建变量
a = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
# 创建占位符
x = tf.placeholder(dtype=tf.float32)

# 创建线性回归模型
linear_model = a*x+b

# 创建理论上的y值即linear_model值
y = tf.placeholder(dtype=tf.float32)
# 创建损失函数
loss = tf.reduce_sum(tf.square(linear_model-y))
# 创建最优控制器
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 训练,使数据所损失的值最小
train = optimizer.minimize(loss)

# 创建训练数据
x_train = [1, 2, 3, 4]
y_train = [7, 8, 9, 10]
# 初始化全局的变量
init = tf.global_variables_initializer()
# 创建会话层
sess = tf.Session()
sess.run(init)
# 训练的迭代次数

for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

cur_a, cur_b, cur_loss = sess.run([a, b, loss],{x: x_train, y: y_train})

print("a= %s b=%s loss=%s"%(cur_a, cur_b, cur_loss))




