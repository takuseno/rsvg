import tensorflow as tf
import numpy as np


def build_train(actor,
                critic,
                obs_dim,
                num_actions,
                batch_size,
                gamma=1.0,
                scope='ddpg',
                tau=0.001,
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        n_episode_ph = tf.placeholder(tf.int32, [], name='n_episode')
        step_size_ph = tf.placeholder(tf.int32, [], name='step_size')

        actor_rnn_state_ph0 = tf.placeholder(
            tf.float32, [None, 64], name='actor_rnn_state0')
        actor_rnn_state_ph1 = tf.placeholder(
            tf.float32, [None, 64], name='actor_rnn_state1')
        actor_rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            actor_rnn_state_ph0, actor_rnn_state_ph1)

        critic_rnn_state_ph0 = tf.placeholder(
            tf.float32, [None, 64], name='critic_rnn_state0')
        critic_rnn_state_ph1 = tf.placeholder(
            tf.float32, [None, 64], name='critic_rnn_state1')
        critic_rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            critic_rnn_state_ph0, critic_rnn_state_ph1)

        # actor network
        policy_t, actor_lstm_state = actor(
            obs_t_input, n_episode_ph, step_size_ph, actor_rnn_state_tuple,
            obs_dim, num_actions, scope='actor')
        actor_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/actor'.format(scope))

        # target actor network
        policy_tp1, _ = actor(
            obs_tp1_input, n_episode_ph, step_size_ph, actor_rnn_state_tuple,
            obs_dim, num_actions, scope='target_actor')
        target_actor_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/target_actor'.format(scope))

        # critic network
        q_t, critic_lstm_state = critic(
            obs_t_input, act_t_ph, n_episode_ph, step_size_ph,
            critic_rnn_state_tuple, obs_dim, num_actions, scope='critic')
        q_t_with_actor, _ = critic(
            obs_t_input, policy_t, n_episode_ph, step_size_ph,
            critic_rnn_state_tuple, obs_dim, num_actions, scope='critic', reuse=True)
        critic_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/critic'.format(scope))

        # target critic network
        q_tp1, _ = critic(
            obs_tp1_input, policy_tp1, n_episode_ph, step_size_ph,
            critic_rnn_state_tuple, obs_dim, num_actions, scope='target_critic')
        target_critic_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/target_critic'.format(scope))

        # loss
        with tf.variable_scope('target_q'):
            v = (1 - done_mask_ph) * gamma * tf.stop_gradient(q_tp1)
            target_q = rew_t_ph + v
        with tf.variable_scope('loss'):
            critic_loss = tf.reduce_mean(
                tf.square(target_q - q_t), name='critic_loss')
            actor_loss = -tf.reduce_mean(q_t_with_actor, name='actor_loss')

        # optimize operations
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_optimize_expr = critic_optimizer.minimize(
            critic_loss, var_list=critic_func_vars)
        actor_optimizer = tf.train.AdamOptimizer(0.0001)
        actor_optimize_expr = actor_optimizer.minimize(
            actor_loss, var_list=actor_func_vars)

        # update critic target operations
        with tf.variable_scope('update_critic_target'):
            update_critic_target_expr = []
            sorted_vars = sorted(
                critic_func_vars, key=lambda v: v.name)
            sorted_target_vars = sorted(
                target_critic_func_vars, key=lambda v: v.name)
            # assign critic variables to target critic variables
            for var, var_target in zip(sorted_vars, sorted_target_vars):
                new_var = tau * var + (1 - tau) * var_target
                update_critic_target_expr.append(var_target.assign(new_var))
            update_critic_target_expr = tf.group(*update_critic_target_expr)

        # update actor target operations
        with tf.variable_scope('update_actor_target'):
            update_actor_target_expr = []
            sorted_vars = sorted(
                actor_func_vars, key=lambda v: v.name)
            sorted_target_vars = sorted(
                target_actor_func_vars, key=lambda v: v.name)
            # assign actor variables to target actor variables
            for var, var_target in zip(sorted_vars, sorted_target_vars):
                new_var = tau * var + (1 - tau) * var_target
                update_actor_target_expr.append(var_target.assign(new_var))
            update_actor_target_expr = tf.group(*update_actor_target_expr)

        def act(obs, rnn_state0, rnn_state1):
            feed_dict = {
                obs_t_input: obs,
                actor_rnn_state_ph0: rnn_state0,
                actor_rnn_state_ph1: rnn_state1,
                n_episode_ph: 1,
                step_size_ph: 1
            }
            return tf.get_default_session().run(
                [policy_t, actor_lstm_state], feed_dict=feed_dict)

        rnn_state_shape = [batch_size, 64]
        init_state= np.zeros(rnn_state_shape, dtype=np.float32)

        def train_actor(obs, n_episode, step_size):
            feed_dict = {
                obs_t_input: obs,
                n_episode_ph: n_episode,
                step_size_ph: step_size,
                actor_rnn_state_ph0: init_state,
                actor_rnn_state_ph1: init_state,
                critic_rnn_state_ph0: init_state,
                critic_rnn_state_ph1: init_state
            }
            loss_val, _ = tf.get_default_session().run(
                [actor_loss, actor_optimize_expr], feed_dict=feed_dict)
            return loss_val

        def train_critic(obs_t, act, rew, obs_tp1, done, n_episode, step_size):
            feed_dict = {
                obs_t_input: obs_t,
                act_t_ph: act,
                rew_t_ph: rew,
                obs_tp1_input: obs_tp1,
                done_mask_ph: done,
                n_episode_ph: n_episode,
                step_size_ph: step_size,
                actor_rnn_state_ph0: init_state,
                actor_rnn_state_ph1: init_state,
                critic_rnn_state_ph0: init_state,
                critic_rnn_state_ph1: init_state
            }
            loss_val, _ = tf.get_default_session().run(
                [critic_loss, critic_optimize_expr], feed_dict=feed_dict)
            return loss_val

        def update_actor_target():
            tf.get_default_session().run(update_actor_target_expr)

        def update_critic_target():
            tf.get_default_session().run(update_critic_target_expr)

        return act, train_actor, train_critic, update_actor_target, update_critic_target
