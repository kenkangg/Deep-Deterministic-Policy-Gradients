import keras
from keras.layers import Dense, MaxPooling2D, Input, merge
from keras.models import Model
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001


class Critic:
    def __init__(self, sess, state_shape, action_shape):
        self.sess = sess
        self.model, self.state_input, self.action_input, self.gradients, self.optimizer, self.y = self.create_network(state_shape, action_shape)

    def create_network(self, state_shape, action_shape):
        """ Create Critic Network """

        state_input = Input(shape=state_shape)
        action_input = Input(shape=action_shape)

        y = tf.placeholder(tf.float32, [None, 1])

        s_dense1 = Dense(64, activation='relu')(state_input)
        s_dense2 = Dense(64, activation='relu')(s_dense1)

        a_dense1 = Dense(64, activation='relu')(action_input)
        a_dense2 = Dense(64, activation='relu')(a_dense1)

        merged = merge([s_dense2, a_dense2], 'sum')

        dense3 = Dense(64, activation = 'relu')(merged)
        output = Dense(1, activation='sigmoid')(dense3)

        model = Model(inputs=[state_input, action_input], outputs=[output])

        cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))

        grad_values = tf.gradients(cost, model.trainable_weights)
        grad_action = tf.gradients(cost, action_input)
        grad_pairs = zip(grad_values, model.trainable_weights)



        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(grad_pairs)
        self.sess.run(tf.global_variables_initializer())
        print('hi')

        return model, state_input, action_input, grad_action, optimizer, y

    def get_gradients(self, state, action, y):
        return self.sess.run(self.gradients, feed_dict={self.state_input:state, self.action_input:action, self.y:y})

    def train(self, state, action, y):
        self.sess.run(self.optimizer, feed_dict={self.state_input:state, self.action_input:action, self.y:y})


    def predict_Q(self, state, action):
        return self.model.predict([state,action])
