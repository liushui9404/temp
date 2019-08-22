# coding:utf-8
import tensorflow as tf

learning_rate = 0.1
state_size = 8 # hidden layer num of features
n_classes = 19
n_features = 4

#输入
x = tf.placeholder(tf.float32, [None, None, n_features], name='input_placeholder')  #batch_size, time_step, feat_len
# y = tf.placeholder(tf.float32, [None, None, n_classes], name='labels_placeholder')  #batch_size, time_step, n_classes

batch_size = tf.placeholder(tf.int32, (), name='batch_size')
time_steps = tf.placeholder(tf.int32, (), name='times_step')

#双向rnn
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)

init_fw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
init_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)



outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                       lstm_bw_cell,
                                                       x,
                                                       initial_state_fw = init_fw,
                                                       initial_state_bw = init_bw)


# weights = tf.get_variable("weights", [2 * state_size, n_classes], dtype=tf.float32,   #注意这里的维度
#                          initializer = tf.random_normal_initializer(mean=0, stddev=1))
# biases = tf.get_variable("biases", [n_classes], dtype=tf.float32, 
#                         initializer = tf.random_normal_initializer(mean=0, stddev=1))

def train_network(num_epochs = 100):
    inputs=[[[1,1,1,1],[2,2,2,2]],[[1,1,1,1],[2,2,2,2]],[[1,1,1,1],[2,2,2,2]]]

    time_step = 2



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #初始化variable
        ans = sess.run(outputs,feed_dict = {x: inputs,
                                       batch_size:len(inputs),
                                       time_steps: time_step})
        
        # output_fw, output_bw = outputs
        # states_fw, states_bw = states
        # intervel = 5
        
        # for epoch in range(num_epochs):
        #     #开始训练
        #     for idx, (time_step, inputs, labels, idxes) in enumerate(get_dump_seq_data(1)):
        #         _= sess.run([train_op],
        #                    feed_dict = {x: inputs,
        #                                y:labels,
        #                                batch_size:len(inputs),
        #                                time_steps: time_step})
        #     print("epoch %d train done" % epoch)
            # #这一轮训练完毕，计算损失值和准确率
            
            # if epoch % intervel == 0 and epoch > 1:
            #     #训练集误差
            #     acc_record, total_df, total_acc, loss = compute_accuracy(sess, 1)  #这里是我自定义的函数，与整个架构关系不大
                #验证集误差
train_network(100)