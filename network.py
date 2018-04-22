import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens,
                        inpt,
                        n_episode,
                        step_size,
                        rnn_state_tuple,
                        obs_dim,
                        num_actions,
                        scope='actor',
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden,
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
            out = tf.nn.relu(out)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
            rnn_input = tf.reshape(out, [n_episode, step_size, 64])
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_input, initial_state=rnn_state_tuple,
                sequence_length=tf.fill([n_episode], step_size),
                time_major=False)
            out = tf.reshape(lstm_outputs, [n_episode * step_size, 64])

        # mean value of normal distribution
        initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        mu = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='mu')
        mu = tf.nn.tanh(mu)

        # variance of normal distribution
        sigma = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='sigma')
        sigma = tf.nn.softplus(sigma)

        # sample actions from normal distribution
        out = tf.squeeze(
            tf.distributions.Normal(mu, sigma).sample(num_actions), [0])
    return out, lstm_state

def _make_critic_network(hiddens,
                         inpt,
                         action,
                         n_episode,
                         step_size,
                         rnn_state_tuple,
                         obs_dim,
                         num_actions,
                         scope='critic',
                         reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens[:-1]:
            out = tf.layers.dense(out, hidden,
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
            out = tf.nn.relu(out)

        # concat action
        out = tf.concat([out, action], axis=1)
        out = tf.layers.dense(out, hiddens[-1],
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
        out = tf.nn.relu(out)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
            rnn_input = tf.reshape(out, [n_episode, step_size, 64])
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_input, initial_state=rnn_state_tuple,
                sequence_length=tf.fill([n_episode], step_size),
                time_major=False)
            out = tf.reshape(lstm_outputs, [n_episode * step_size, 64])

        initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        out = tf.layers.dense(out, 1, kernel_initializer=initializer)
    return out, lstm_state

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network(hiddens):
    return lambda *args, **kwargs: _make_critic_network(hiddens, *args, **kwargs)
