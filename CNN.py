import tensorflow as tf
import numpy as np
import pickle
import time
import data_utils

from sklearn.model_selection import train_test_split



'''


1. convolutional neural netowrk without embedding layers


'''

x_train_data_path = 'processed_data/x_train_data.p'
y_train_data_path = 'processed_data/y_train_data.p'
train_data_parameter = 'processed_data/x_y_parameter.p'
query_dir = 'dataset/programs_200/'

tensorboard_data_path = './logs/CNN/'

epoch_num = 2
batch_size = 128
learning_rate = 0.002
context_size = 10
hidden_size = 128

class Code_Completion_Model(object):

    def __init__(self, x_data, y_data, token_set, string2int, int2string):
        batch_num = len(x_data) // batch_size
        x_data, y_data = np.array(x_data[:batch_num * batch_size]), np.array(y_data[:batch_num * batch_size])
        self.reshape_data(x_data, y_data)
        self.x_data, self.valid_x, self.y_data, self.valid_y = \
            train_test_split(x_data, y_data, train_size=0.9)
        self.data_size = len(self.x_data)
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
        self.re_input_x = tf.reshape(self.input_x, [-1, self.tokens_size, 1, 1])
        self.re_output_y = tf.reshape(self.output_y, [-1, self.tokens_size, 1, 1])

        weights = {'h1': tf.Variable(tf.truncated_normal(shape=[self.tokens_size, hidden_size])),
                   'h2': tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size])),
                   'output': tf.Variable(tf.truncated_normal(shape=[hidden_size, self.tokens_size]))}
        biases = {'h1': tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'h2': tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'output': tf.Variable(tf.constant(0.1, shape=[self.tokens_size], dtype=tf.float32))}

        conv1_layer = tf.nn.conv2d(self.re_input_x, weights['conv1'], strides=[1,1,1,1], padding='SAME')
        pool1_layer = tf.nn.avg_pool(conv1_layer, [2,2], strides=[1,1,1,1],padding='VALID')
        relu1_layer = tf.nn.relu(pool1_layer)

        conv2_layer = tf.nn.conv2d(relu1_layer, weights['conv2'], strides=[1,1,1,1], padding='SAME')
        pool2_layer = tf.nn.avg_pool(conv2_layer, ksize=[2,2], strides=[1,1,1,1], padding='VALID')
        relu2_layer = tf.nn.relu(pool2_layer)

        h1_layer = tf.matmul(relu2_layer, weights['h1']) + biases['h1']
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
        writer = tf.summary.FileWriter(tensorboard_data_path, self.sess.graph)
        time_begin = time.time()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            batch_generator = self.get_batch()
            for i in range(0, len(self.x_data), batch_size):
                batch_x, batch_y = next(batch_generator)
                feed = {self.input_x: batch_x, self.output_y: batch_y}
                _, summary_str = self.sess.run([self.optimizer_op, self.merged], feed_dict=feed)
                writer.add_summary(summary_str, epoch*self.data_size + i)
                writer.flush()
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
        previous_token_list = prefix[-context_size:]
        context_representation = np.zeros(self.tokens_size)

        for token in previous_token_list:
            prev_token_string = data_utils.token_to_string(token)
            pre_token_x = data_utils.one_hot_encoding(prev_token_string, self.string_to_index)
            context_representation += np.array(pre_token_x)

        feed = {self.input_x: [context_representation]}
        prediction = self.sess.run(self.prediction, feed)[0]
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