# coding:utf-8
import tensorflow as tf

learning_rate = 0.1
state_size = 16 # hidden layer num of features

n_features = 4
batch_size = 2
max_time_steps = 5
max_pair_num = 20      # p,m的最大数量
sequence_length = [4,5]

#输入
x = tf.placeholder(tf.float32, [batch_size, max_time_steps, n_features], name='input_placeholder')  #batch_size, time_step, feat_len
# y = tf.placeholder(tf.float32, [None, None, n_classes], name='labels_placeholder')  #batch_size, time_step, n_classes



#双向rnn
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)

init_fw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
init_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)



outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                       lstm_bw_cell,
                                                       x,
                                                       sequence_length,
                                                       initial_state_fw = init_fw,
                                                       initial_state_bw = init_bw)


# weights = tf.get_variable("weights", [2 * state_size, n_classes], dtype=tf.float32,   #注意这里的维度
#                          initializer = tf.random_normal_initializer(mean=0, stddev=1))
# biases = tf.get_variable("biases", [n_classes], dtype=tf.float32, 
#                         initializer = tf.random_normal_initializer(mean=0, stddev=1))
fw,bw = outputs
final_fw,final_bw = final_states

p_index = [[2],[3]]
mention_index = [[2,1],[3,2]]
labels = [[2,1],[3,2]]   # 表示第一句的第2,1成指代关系,第二句的3,2成指代关系
sentence_length = [4,5]
def buildFFNNinput(fw,bw,p_index,mention_index,max_time_steps,labels):
    p_list = []
    m_list = []
    batch_list = []
    temp = []
    pair_labels = []   # 告诉神经网络是哪个p与mention连接,用于计算损失时
    temp_labels = []

    for i in range(batch_size):
        for p in p_index[i]:
            p_represent = tf.concat([fw[i][p],bw[i][p]],0)
            for m in mention_index[i]:
                m_represent = tf.concat([fw[i][m],bw[i][m]],0)
                temp.append(tf.concat([p_represent,m_represent],0))
                # print(labels[i][1])
                if p==labels[i][0] and m==labels[i][1]:
                    temp_labels.append(1)
                else:
                    temp_labels.append(0)
        batch_list.append(temp)
        pair_labels.append(temp_labels)
        temp = []
        temp_labels = []

    return batch_list,pair_labels

def pair_norm(batch_list,pair_labels,max_pair_num,state_size):
    '''
    将batch_list变规整
    '''
    padding = tf.zeros([4*state_size])
    len_pairs = []
    for i in range(len(batch_list)):
        len_pair = len(batch_list[i])
        len_pairs.append(len_pair)
        for j in range(max_pair_num):
            if j<len_pair:
                pass
            else:
                batch_list[i].append(padding)
                pair_labels[i].append(-1)
    return batch_list,pair_labels,len_pairs

def buildFFNN(batch_list,state_size):
    ffnn_input = tf.reshape(batch_list,[-1,4*state_size])
    weights = tf.get_variable("weights", [4 * state_size, 1], dtype=tf.float32,   #注意这里的维度
                         initializer = tf.random_normal_initializer(mean=0, stddev=1))
    
    biases = tf.get_variable("biases", [1], dtype=tf.float32, 
                        initializer = tf.random_normal_initializer(mean=0, stddev=1))
    
    scores = tf.nn.bias_add(tf.nn.relu(tf.matmul(ffnn_input,weights)),biases)
    scores = tf.reshape(scores,[batch_size,-1])
    return scores


def cal_loss(scores,pair_labels,len_pairs,max_pair_num):
    loss = 0
    for i in range(batch_size):
        scores = tf.strided_slice(scores,[0],[len_pairs[i]])
        pair_labels = tf.strided_slice(pair_labels,[0],[len_pairs[i]])
        
        loss += -1*tf.reduce_sum(tf.multiply(tf.cast(pair_labels,tf.float32), tf.log(scores)))
    return loss


def optimizer(loss):
    rain_step = tf.train.AdamOptimizer(0.00001).minimize(loss)


        


batch_list,pair_labels = buildFFNNinput(fw,bw,p_index,mention_index,max_time_steps,labels)
# print("batch_list",len(batch_list))
batch_list,pair_labels,len_pairs = pair_norm(batch_list,pair_labels,max_pair_num,state_size)
scores = buildFFNN(batch_list,state_size)
loss = cal_loss(scores,pair_labels,len_pairs,max_pair_num)
optimizer(loss)

def train_network(num_epochs = 1):
    inputs=[[[1,1,1,1],[2,2,2,2],[3,3,3,3],[1,1,1,0],[0,0,0,0]],[[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2],[5,5,5,5]]]




    with tf.Session() as sess:
        for epoch in range(num_epochs):
            sess.run(tf.global_variables_initializer())  #初始化variable
            ans = sess.run(loss,feed_dict = {x: inputs})
            print(ans)
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