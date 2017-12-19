import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens, inpt, n_episode, step_size,
        rnn_state_tuple, num_actions, scope='actor', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden,
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
            out = tf.nn.relu(out)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
            print(out)
            rnn_input = tf.reshape(
                out,
                [n_episode, step_size, tf.cast(inpt.get_shape()[1], tf.int32)]
            )
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_input,
                initial_state=rnn_state_tuple,
                sequence_length=step_size,
                time_major=False
            )
            out = tf.reshape(lstm_outputs, [n_episode * step_size, 64])

        out = tf.layers.dense(
            out,
            num_actions,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3,
                maxval=3e-3
            )
        )
        out = tf.nn.tanh(out)
    return out, lstm_state

def _make_critic_network(inpt, action, n_episode, step_size,
        rnn_state_tuple, num_actions, scope='critic', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = tf.layers.dense(inpt, 64,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
        out = tf.nn.relu(out)

        out = tf.concat([out, action], axis=1)
        out = tf.layers.dense(out, 64,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
        out = tf.nn.relu(out)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
            rnn_input = tf.reshape(
                out,
                [n_episode, step_size, inpt.get_shape()[1]]
            )
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_input,
                initial_state=rnn_state_tuple,
                sequence_length=step_size,
                time_major=False
            )
            out = tf.reshape(lstm_outputs, [n_episode * step_size, 64])

        out = tf.layers.dense(
            out,
            1,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3,
                maxval=3e-3
            )
        )
    return out, lstm_state

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network():
    return lambda *args, **kwargs: _make_critic_network(*args, **kwargs)
