import tensorflow as tf


from data_generator import DataGenerator
from setting import Setting

embed_setting = Setting()
show_every_n = embed_setting.show_every_n
save_every_n = embed_setting.save_every_n
model_save_path = 'trained_model/node2vector/node2vec.model'


class NodeToVec_NT(object):

    def __init__(self, num_ntoken, num_ttoken,
                 embed_dim=300,
                 learning_rate=0.001,
                 n_sampled=100,
                 num_epochs=1,
                 alpha = 0.7):
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.n_sampled = n_sampled
        self.num_epochs = num_epochs
        self.alpha = 0.7

    def build_input(self):
        input_token = tf.placeholder(tf.int32, [None])
        nt_target = tf.placeholder(tf.int32, [None, 1])
        tt_target = tf.placeholder(tf.int32, [None, 1])
        return input_token, nt_target, tt_target

    def build_embedding(self, token):
        with tf.name_scope('embedding_matrix'):
            embedding_matrix = tf.Variable(
                tf.float32, tf.truncated_normal([self.num_ttoken, self.embed_dim]))
            embedding_rep = tf.nn.embedding_lookup(embedding_matrix, token)
        return embedding_rep

    def build_nt_output(self, input_):
        nt_weight = tf.get_variable('nt_weight', [self.num_ntoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        nt_bias = tf.get_variable('nt_bias', [self.num_ntoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        output = tf.matmul(nt_weight, input_) + nt_bias
        return output

    def build_tt_weight(self):
        tt_weight = tf.get_variable('tt_weight', [self.num_ttoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        tt_bias = tf.get_variable('tt_bias', [self.num_ttoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        return tt_weight, tt_bias

    def build_nt_loss(self, logits, targets):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=targets)
        loss = tf.reduce_mean(loss)
        return loss

    def build_tt_loss(self, weight, bias, embed_input, target):
        loss = tf.nn.sampled_softmax_loss(
            weight, bias, target, embed_input, self.n_sampled, self.num_ttoken)
        loss = tf.reduce_mean(loss)
        return loss

    def build_loss(self, nt_loss, tt_loss):
        nt_loss = self.alpha * nt_loss
        tt_loss = (1 - self.alpha) * tt_loss
        loss = tf.add(nt_loss, tt_loss)
        return loss

    def build_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_model(self):
        tf.reset_default_graph()
        self.input_token, self.nt_target, self.tt_target = self.build_input()
        self.embed_inputs = self.build_embedding(self.input_token)
        nt_logits = self.build_nt_output(self.embed_inputs)
        tt_weight, tt_bias = self.build_tt_weight()
        self.nt_loss = self.build_nt_loss(nt_logits, self.nt_target)
        self.tt_loss = self.build_tt_loss(tt_weight, tt_bias, self.embed_inputs, self.tt_target)
        self.loss = self.build_loss(self.nt_loss, self.tt_loss)
        self.optimizer = self.build_optimizer(self.loss)

    def train(self):
        global_step = 0
        saver = tf.train.Saver(max_to_keep=self.num_epochs + 1)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        generator = DataGenerator()
        for epoch in range(self.num_epochs):
            sub_data = generator.get_embedding_sub_data()
            for data in sub_data:
                batch_generator = generator.get_embedding_batch(data)
                for batch_tt_x, batch_nt_y, batch_tt_y in batch_generator:
                    global_step += 1
                    feed = {
                        self.input_token:batch_tt_x,
                        self.nt_target:batch_nt_y,
                        self.tt_target:batch_tt_y,
                    }
                    n_loss, t_loss, loss, _ = session.run([
                        self.nt_loss, self.tt_loss, self.loss, self.optimizer], feed_dict=feed)

                    if global_step % show_every_n == 0:
                        train_info = 'epoch:{} '.format(epoch) + \
                            'step:{} '.format(global_step) + \
                            'loss(n+t):{} ({} + {}) '.format(loss, n_loss, t_loss)
                        print(train_info)

            print('epoch{} model saved...'.format(epoch))
            saver.save(session, save_path=model_save_path)

        session.close()







class NodeToVec_TT(object):

    def __init__(self, num_ntoken, num_ttoken,
                 embed_dim=300,
                 learning_rate=0.001,
                 n_sampled=100,
                 num_epochs=1):
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.n_sampled = n_sampled
        self.num_epochs = num_epochs

    def build_input(self):
        input_token = tf.placeholder(tf.int32, [None])
        nt_target = tf.placeholder(tf.int32, [None, 1])
        tt_target = tf.placeholder(tf.int32, [None, 1])
        return input_token, nt_target, tt_target

    def build_embedding(self, token):
        with tf.name_scope('embedding_matrix'):
            embedding_matrix = tf.Variable(tf.float32, tf.truncated_normal([self.num_ttoken, self.embed_dim]))
            embedding_rep = tf.nn.embedding_lookup(embedding_matrix, token)
        return embedding_rep

    def build_nt_output(self, input_):
        nt_weight = tf.get_variable('nt_weight', [self.num_ntoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        nt_bias = tf.get_variable('nt_bias', [self.num_ntoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        output = tf.matmul(nt_weight, input_) + nt_bias
        return output

    def build_tt_weight(self):
        tt_weight = tf.get_variable('tt_weight', [self.num_ttoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        tt_bias = tf.get_variable('tt_bias', [self.num_ttoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        return tt_weight, tt_bias

    def build_nt_loss(self, logits, targets):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=targets)
        loss = tf.reduce_mean(loss)
        return loss

    def build_tt_loss(self, weight, bias, embed_input, target):
        loss = tf.nn.sampled_softmax_loss(weight, bias, target, embed_input, self.n_sampled, self.num_ttoken)
        loss = tf.reduce_mean(loss)
        return loss

    def build_loss(self, nt_loss, tt_loss):
        loss = tf.add(nt_loss, tt_loss)
        return loss

    def build_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_model(self):
        tf.reset_default_graph()
        self.input_token, self.nt_target, self.tt_target = self.build_input()
        self.embed_inputs = self.build_embedding(self.input_token)
        nt_logits = self.build_nt_output(self.embed_inputs)
        tt_weight, tt_bias = self.build_tt_weight()
        self.nt_loss = self.build_nt_loss(nt_logits, self.nt_target)
        self.tt_loss = self.build_tt_loss(tt_weight, tt_bias, self.embed_inputs, self.tt_target)
        self.loss = self.build_loss(self.nt_loss, self.tt_loss)
        self.optimizer = self.build_optimizer(self.loss)

    def train(self):
        global_step = 0
        saver = tf.train.Saver(max_to_keep=self.num_epochs + 1)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        generator = DataGenerator()
        for epoch in range(self.num_epochs):
            sub_data = generator.get_embedding_sub_data()
            for data in sub_data:
                batch_generator = generator.get_embedding_batch(data)
                for batch_tt_x, batch_nt_y, batch_tt_y in batch_generator:
                    global_step += 1
                    feed = {
                        self.input_token:batch_tt_x,
                        self.nt_target:batch_nt_y,
                        self.tt_target:batch_tt_y,
                    }
                    n_loss, t_loss, loss, _ = session.run([
                        self.nt_loss, self.tt_loss, self.loss, self.optimizer], feed_dict=feed)

                    if global_step % show_every_n == 0:
                        train_info = 'epoch:{} '.format(epoch) + \
                            'step:{} '.format(global_step) + \
                            'loss(n+t):{} ({} + {}) '.format(loss, n_loss, t_loss)
                        print(train_info)

            print('epoch{} model saved...'.format(epoch))
            saver.save(session, save_path=model_save_path)

        session.close()