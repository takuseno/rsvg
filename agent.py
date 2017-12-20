import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf


class Agent(object):
    def __init__(self, actor, critic, obs_dim,
                num_actions, replay_buffer, batch_size=4, gamma=0.9):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.last_obs = None
        self.t = 0
        self.replay_buffer = replay_buffer
        self.actor_lstm_state = np.zeros((2, 1, 64), dtype=np.float32)

        self._act,\
        self._train_actor,\
        self._train_critic,\
        self._update_actor_target,\
        self._update_critic_target = build_graph.build_train(
            actor=actor,
            critic=critic,
            obs_dim=obs_dim,
            num_actions=num_actions,
            gamma=gamma
        )

    def act(self, obs):
        return self._act([obs])[0]

    def act_and_train(self, obs, reward, episode):
        action, self.actor_lstm_state  = self._act(
            [obs],
            self.actor_lstm_state[0],
            self.actor_lstm_state[1]
        )
        action = np.clip(action[0], -2, 2)
        print(action)
        reward /= 10.0

        if self.t > 10 * 200:
            # sample experiences
            obs_t,\
            actions,\
            rewards,\
            obs_tp1,\
            dones = self.replay_buffer.sample_sequences(self.batch_size, 8)

            # reshape data to feed network
            obs_t = np.reshape(obs_t, (-1, self.obs_dim))
            actions = np.reshape(actions, (-1, self.num_actions))
            rewards = np.reshape(rewards, (-1))
            obs_tp1 = np.reshape(obs_tp1, (-1, self.obs_dim))
            dones = np.reshape(dones, (-1))

            # update networks
            actor_error = self._train_actor(obs_t, self.batch_size, 8)
            critic_error = self._train_critic(
                obs_t,
                actions,
                rewards,
                obs_tp1,
                dones,
                self.batch_size,
                8
            )

            # update target networks
            self._update_actor_target()
            self._update_critic_target()

        if self.last_obs is not None:
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=False
            )

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.replay_buffer.append(
            obs_t=self.last_obs,
            action=self.last_action,
            reward=reward,
            obs_tp1=obs,
            done=done
        )
        self.replay_buffer.end_episode()
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = []
        self.actor_lstm_state = np.zeros((2, 1, 64), dtype=np.float32)
