import tensorflow as tf
import numpy as np
import pickle
import time
import data_utils

from sklearn.model_selection import train_test_split



'''
使用TensorFlow自带的layers构建基本的神经网络对token进行预测，
可以声明使用多少个context tokens 进行预测

多个previous token输入神经网络的方法有两种想法：
1. 将每个token的representation vector相连，合成一个大的vector输入到神经网络，
    所以说神经网络的输入层大小应为：每个token vector length * number of previous token
2. 应为目前表示每个token 使用的方法为one hot encoding，也就是说对每个token都是有且仅有一位为1，其余位为0，
    所以可以考虑直接将所有的previous token相加，这样做的好处是NN输入层大小永远等于vector length。缺点是没有理论依据，不知道效果是否会更好


1. concatenate the representations of previous tokens to a huge vector representation
2. add the representations of previous tokens together


'''

x_train_data_path = 'processed_data/x_train_data.p'
y_train_data_path = 'processed_data/y_train_data.p'
train_data_parameter = 'processed_data/x_y_parameter.p'
query_dir = 'dataset/programs_200/'

tensorboard_data_path = './logs/MultiContext'

epoch_num = 1
batch_size = 64
learning_rate = 0.002
context_size = 10
hidden_size = 128

class Code_Completion_Model:

    def __init__(self, x_data, y_data, token_set, string2int, int2string):
        batch_num = len(x_data) // batch_size
        x_data, y_data = np.array(x_data[:batch_num * batch_size]), np.array(y_data[:batch_num * batch_size])
        self.reshape_data(x_data, y_data)
        self.x_data, self.valid_x, self.y_data, self.valid_y = \
            train_test_split(x_data, y_data, train_size=0.9)
        self.index_to_string = int2string
        self.string_to_index = string2int
        self.tokens_set = token_set
        self.tokens_size = len(token_set)

    def reshape_data(self, x_data, y_data):
        x = []
        y = []
        for index, token in enumerate(x_data):
            if index >= context_size - 1:
                tempTokens = np.sum(x_data[index - context_size + 1:index + 1, :], axis=0)
                x.append(tempTokens)
                y.append(y_data[index])
        return x, y;

    # neural network functions
    def create_NN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size], name='input_x')
        self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size], name='output_y')
        weights = {'h1': tf.Variable(tf.truncated_normal(shape=[self.tokens_size, hidden_size])),
                   'h2': tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size])),
                   'output': tf.Variable(tf.truncated_normal(shape=[hidden_size, self.tokens_size]))}
        biases = {'h1': tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'h2': tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'output': tf.Variable(tf.constant(0.1, shape=[self.tokens_size], dtype=tf.float32))}

        h1_layer = tf.matmul(self.input_x, weights['h1']) + biases['h1']
        h1_layer = tf.nn.relu(h1_layer)
        h2_layer = tf.matmul(h1_layer, weights['h2']) + biases['h2']
        h2_layer = tf.nn.relu(h2_layer)
        output_layer = tf.matmul(h2_layer, weights['output']) + biases['output']
        self.prediction = tf.argmax(output_layer, 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=self.output_y)
        self.loss = tf.reduce_mean(loss)
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        equal = tf.equal(tf.argmax(output_layer, 1), tf.argmax(self.output_y, 1))
        accuracy = tf.cast(equal, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy)

        tf.summary.histogram('weight1', weights['h1'])
        tf.summary.histogram('weight2', weights['h2'])
        tf.summary.histogram('output_weight', weights['output'])
        tf.summary.histogram('bias1', biases['h1'])
        tf.summary.histogram('bias2', biases['h2'])
        tf.summary.histogram('output_bias', biases['output'])
        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def get_batch(self):
        for i in range(0, len(self.x_data), batch_size):
            batch_x = self.x_data[i:i + batch_size];
            batch_y = self.y_data[i:i + batch_size];
            yield batch_x, batch_y

    def train(self):
        self.create_NN()
        self.sess = tf.Session()
        time_begin = time.time()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            batch_generator = self.get_batch()
            for i in range(0, len(self.x_data), batch_size):
                batch_x, batch_y = next(batch_generator)
                feed = {self.input_x: batch_x, self.output_y: batch_y}
                self.sess.run(self.optimizer_op, feed_dict=feed)
                if (i // batch_size) % 2000 == 0:
                    show_loss, show_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed)
                    print('epoch: %d, training_step: %d, loss: %.2f, accuracy:%.3f' % (epoch, i, show_loss, show_acc))
        time_end = time.time()
        print('training time cost: %.3f s' % (time_end - time_begin))

    # query test
    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole. In this function, use only one token before hole token to predict
        '''
        previous_token_list = prefix[-previous_token_num:]
        context_representation = np.zeros(self.tokens_size)

        for token in previous_token_list:
            prev_token_string = data_utils.token_to_string(token)
            pre_token_x = data_utils.one_hot_encoding(prev_token_string, self.string_to_index)
            context_representation += np.array(pre_token_x)

        feed = {self.input_x: [context_representation]}
        prediction = self.sess.run(self.prediction_index, feed)[0]
        best_string = self.index_to_string[prediction]
        best_token = data_utils.string_to_token(best_string)
        return [best_token]

    # test model
    def test_model(self, query_test_data):
        correct = 0.0
        correct_token_list = []
        incorrect_token_list = []
        for token_sequence in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(token_sequence)
            prediction = self.query_test(prefix, suffix)[0]
            if data_utils.token_equals([prediction], expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(query_test_data)
        return accuracy


if __name__ == '__main__':

    x_train_data_path = 'processed_data/x_train_data.p'
    y_train_data_path = 'processed_data/y_train_data.p'
    train_data_parameter = 'processed_data/x_y_parameter.p'
    x_data = data_utils.load_data_with_pickle(x_train_data_path)
    y_data = data_utils.load_data_with_pickle(y_train_data_path)
    token_set, string2int, int2string = data_utils.load_data_with_pickle(train_data_parameter)


    #model train
    model = Code_Completion_Model(x_data, y_data, token_set, string2int, int2string)
    model.train()

    # test model
    query_test_data = data_utils.load_data_with_file(query_dir)
    accuracy = model.test_model(query_test_data)
    print('query test accuracy: ', accuracy)