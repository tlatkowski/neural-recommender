import tensorflow as tf

from models.base_model import BaseRecommenderModel


class NeuralCollaborativeFiltering(BaseRecommenderModel):

    def __init__(self, num_users, num_items, user_embedding_size=50, item_embedding_size=50, learning_rate=0.001):
        super().__init__(num_users, num_items, user_embedding_size, item_embedding_size, learning_rate)

    def model_implementation(self, users_embedded, items_embedded):
        layers_sizes = [200, 100]
        with tf.name_scope('ncf'):
            users_items = tf.concat([users_embedded, items_embedded], axis=1)

            output = users_items
            for i, layer_size in enumerate(layers_sizes):
                output = tf.layers.dense(output, units=layers_sizes[i], activation=tf.nn.relu)

            prediction = tf.squeeze(tf.layers.dense(output, units=1, activation=tf.nn.sigmoid))
        return prediction
