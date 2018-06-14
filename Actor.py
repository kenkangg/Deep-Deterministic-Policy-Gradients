import keras
from keras.layers import Dense, MaxPooling2D, Input
from keras.models import Model
import keras.backend as K
import tensorflow as tf


LEARNING_RATE = 0.01


class Actor:
    def __init__(self, sess, input_shape, output_size):
        """ Actor Network:

            Input: Observation
            Output: Action (Policy)
        """
        self.sess = sess
        self.model, self.weights, self.input = self.create_network(input_shape, output_size)
        self.optimizer, self.critic_grad, self.gradients = self.get_optimizer(self.model.output, self.weights, output_size)




    def create_network(self, input_shape, output_size):
        """ Create Actor Network """

        input1 = Input(shape=input_shape)
        dense1 = Dense(64, activation='relu')(input1)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs=[input1], outputs=[output])

        return model, model.trainable_weights, input1

    def get_optimizer(self, output, weight_variables, output_size):
        """ Return Optimizer after applying gradients from model """
        out_gradient = tf.placeholder(tf.float32, [None, output_size[0]]) # Holds gradients of output
        grad_values = tf.gradients(output, weight_variables, grad_ys=out_gradient)
        grad_pairs = zip(grad_values, weight_variables)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(grad_pairs)
        self.sess.run(tf.global_variables_initializer())
        return optimizer, out_gradient, grad_values

    def get_gradients(self, state_input, critic_grad):
        return self.sess.run([self.gradients], feed_dict={self.input:state_input, self.critic_grad:critic_grad})

    def train(self, state_input,critic_grad):
        _ = self.sess.run([self.optimizer], feed_dict={self.input:state_input, self.critic_grad:critic_grad})

    def predict_policy(self, x):
        return self.model.predict(x)








        #optimizer, apply gradients (gradients, variables)
