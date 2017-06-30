import tensorflow as tf


class LSTM(object):
    """docstring for LSTM"""

    def __init__(self, n_batch, n_step, n_input, n_output, n_cell, lr=0.006):
        super(LSTM, self).__init__()
        self.batch = n_batch
        self.step = n_step
        self.n_input = n_input
        self.n_output = n_output
        self.cellnum = n_cell
        self.lr = lr
        self.x = tf.placeholder(tf.float32, [None, n_step, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_output])
        self.add_in_layer()
        self.add_lstm_layer()
        self.add_out_layer()
        self.get_loss()
        self.train()
        self.evaluate()

    def add_in_layer(self):
        x_input = tf.reshape(self.x, [-1, self.n_input])
        layer_input = tf.layers.dense(x_input, self.cellnum)
        self.layer_input = tf.reshape(
            layer_input, [-1, self.step, self.cellnum])

    def add_lstm_layer(self):
        self.cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.cellnum)
        self.init_state = self.cell.zero_state(self.batch, dtype=tf.float32)
        self.output, self.state = tf.nn.dynamic_rnn(
            self.cell, self.layer_input, initial_state=self.init_state, time_major=False)

    def add_out_layer(self):
        self.layer_output = tf.layers.dense(self.state[1], self.n_output)

    def get_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.layer_output))

    def train(self):
    	self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def evaluate(self):
    	self.acc = tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.layer_output, 1)), tf.float32)
    	