import tensorflow as tf

from models.base_model import BaseRecommenderModel


class MatrixFactorization(BaseRecommenderModel):

    def __init__(self, num_users, num_items, user_embedding_size=50, item_embedding_size=50, learning_rate=0.001):
        super().__init__(num_users, num_items, user_embedding_size, item_embedding_size, learning_rate)

    def model_implementation(self, users_embedded, items_embedded):
        with tf.name_scope('factorization'):
            users_items = tf.multiply(users_embedded, items_embedded)
            prediction = tf.layers.dense(users_items, units=1, activation=tf.nn.sigmoid)
        return prediction