import gym
import tensorflow as tf
from Actor import Actor
from Critic import Critic
import numpy as np

import random

EPISODES = 20000
EP_DURATION = 100
GAMMA = 0.95


def train_minibatch(minibatch, actor, critic):
    # for sample in minibatch:
    #     state, action, reward, next_state = sample
    #     next_state = np.expand_dims(next_state, 0)
    #
    #     next_action = actor.predict_policy(next_state)[0]
    #     y = reward + GAMMA * critic.predict_Q(next_state, next_action)
    #     critic.train(state, action, y)

    minibatch = np.vstack(minibatch)
    state = np.vstack(minibatch[:,0])
    action = np.vstack(minibatch[:,1])
    reward = np.vstack(minibatch[:,2])
    next_state = np.vstack(minibatch[:,3])
    next_action = actor.predict_policy(next_state)
    y = np.add(reward,GAMMA * critic.predict_Q(next_state, next_action))
    critic.train(state, action, y)


    # action = actor.predict_policy(state)
    # tf.matmul(actor_grad[0] * critic_grad[0])
    critic_grad = critic.get_gradients(state, action, y)
    actor_grad = actor.get_gradients(state, critic_grad[0])
    print(actor_grad[0])

    actor.train(state, critic_grad[0] * actor_grad)



if __name__ == "__main__":

    # Mountain Car Continuous Environment:
    #   Observation Space = [Position, Velocity]
    #   Action Space = [Push Value (range: [-1,1])]
    env = gym.make('MountainCarContinuous-v0')

    sess = tf.Session()

    state_space = env.observation_space.shape
    action_space = env.action_space.shape

    actor = Actor(sess, state_space, action_space)
    critic = Critic(sess, state_space, action_space)

    replay_buffer = []

    for episode in range(EPISODES):
        state = env.reset()

        for t in range(EP_DURATION):
            if episode % 50 == 0:

                env.render()

            state = np.expand_dims(state, 0)

            ### SELECT ACTION FROM ACTOR NETWORK
            action = actor.predict_policy(state)


            ### PERFORM ACTION, OBSERVE REWARD/NEW STATE
            next_state, reward, done, info = env.step(action[0])

            ### STORE TRANSITION IN REPLAY BUFFER
            replay_buffer.append((state, action, reward, next_state))
            state = next_state

            # End Episode if done returns true
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        minibatch = [replay_buffer[i] for i in sorted(random.sample(range(len(replay_buffer)), 10))]
        minibatch = np.array(minibatch)
        train_minibatch(minibatch, actor, critic)
        replay_buffer = []

        print("Episode {} Complete".format(episode + 1))
