import tensorflow as tf


from data_generator import DataGenerator
from setting import Setting

embed_setting = Setting()
show_every_n = embed_setting.show_every_n
save_every_n = embed_setting.save_every_n
num_nt_token = embed_setting.num_non_terminal
num_tt_token = embed_setting.num_terminal

model_save_path = 'trained_model/node2vector/node2vec.model'
nt_train_pair_dir = 'js_dataset/train_pair_data/nt_train_pair/'
tt_train_pair_dir = 'js_dataset/train_pair_data/tt_train_pair/'


class NodeToVec_NT(object):

    def __init__(self, num_ntoken, num_ttoken,
                 embed_dim=300,
                 learning_rate=0.001,
                 n_sampled=100,
                 num_epochs=1,
                 time_steps=80,
                 alpha = 0.7):
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.n_sampled = n_sampled
        self.num_epochs = num_epochs
        self.time_steps = time_steps
        self.alpha = alpha

        self.build_model()

    def build_input(self):
        input_token = tf.placeholder(tf.int32, [None])
        nt_target = tf.placeholder(tf.int32, [None, 1])
        tt_target = tf.placeholder(tf.int32, [None, 1])
        return input_token, nt_target, tt_target

    def build_embedding(self, input_token):
        with tf.name_scope('embedding_matrix'):
            embedding_matrix = tf.Variable(
                tf.truncated_normal([self.num_ntoken, self.embed_dim]), dtype=tf.float32)
            embedding_rep = tf.nn.embedding_lookup(embedding_matrix, input_token)
        return embedding_rep

    def build_nt_output(self, input_):
        nt_weight = tf.get_variable('nt_weight', [self.embed_dim, self.num_ntoken],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        nt_bias = tf.get_variable('nt_bias', [self.num_ntoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        output = tf.matmul(input_, nt_weight) + nt_bias
        return output

    def bulid_onehot_nt_target(self, nt_target):
        onehot_n_target = tf.one_hot(nt_target, self.num_ntoken)
        n_shape = (self.batch_size * self.time_steps, self.num_ntoken)
        #t_shape = (self.batch_size * self.time_steps, self.num_ttoken)
        onehot_n_target = tf.reshape(onehot_n_target, n_shape)
        #onehot_t_target = tf.one_hot(t_target, self.num_ttoken)
        #onehot_t_target = tf.reshape(onehot_t_target, t_shape)
        return onehot_n_target #, onehot_t_target


    def build_nt_loss(self, embed_input, targets):
        nt_weight = tf.get_variable('nt_weight', [self.num_ntoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        nt_bias = tf.get_variable('nt_bias', [self.num_ntoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        loss = tf.nn.sampled_softmax_loss(
            nt_weight, nt_bias, targets, embed_input, self.num_ntoken, self.num_ntoken)
        loss = tf.reduce_mean(loss)
        return loss

    def build_tt_loss(self, embed_input, target):
        tt_weight = tf.get_variable('tt_weight', [self.num_ttoken, self.embed_dim],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        tt_bias = tf.get_variable('tt_bias', [self.num_ttoken], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        loss = tf.nn.sampled_softmax_loss(
            tt_weight, tt_bias, target, embed_input, self.n_sampled, self.num_ttoken)
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
        #nt_logits = self.build_nt_output(self.embed_inputs)
        #self.nt_loss = self.build_nt_loss(nt_logits, self.nt_target)
        self.nt_loss = self.build_nt_loss(self.embed_inputs, self.nt_target)
        self.tt_loss = self.build_tt_loss(self.embed_inputs, self.tt_target)
        self.loss = self.build_loss(self.nt_loss, self.tt_loss)
        self.optimizer = self.build_optimizer(self.loss)

    def train(self):
        global_step = 0
        saver = tf.train.Saver(max_to_keep=self.num_epochs + 1)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        generator = DataGenerator()
        for epoch in range(self.num_epochs):
            data_gen = generator.get_embedding_sub_data(cate='nt')
            for sub_data in data_gen:
                batch_generator = generator.get_embedding_batch(sub_data)
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



if __name__ == '__main__':
    model = NodeToVec_NT(num_nt_token, num_tt_token)
    model.train()