import tensorflow as tf

from models.base_model import BaseRecommenderModel


class AttentionCollaborativeFiltering(BaseRecommenderModel):

    def __init__(self, num_users, num_items, user_embedding_size=50, item_embedding_size=50, learning_rate=0.001):
        super().__init__(num_users, num_items, user_embedding_size, item_embedding_size, learning_rate)

    def model_implementation(self, users_embedded, items_embedded):
        layers_sizes = [200, 100]
        with tf.name_scope('acf'):
            users_self_attention, _ = multiplicative_attention(users_embedded, users_embedded, users_embedded)
            items_self_attention, _ = multiplicative_attention(items_embedded, items_embedded, items_embedded)
            users_items_attention, _ = multiplicative_attention(users_embedded, items_embedded,
                                                                items_embedded)  # TODO check correctness
            users_items = tf.concat([users_self_attention, items_self_attention, users_items_attention], axis=0)

            output = users_items
            for i, layer_size in enumerate(layers_sizes):
                output = tf.layers.dense(output, units=layers_sizes[i], activation=tf.nn.relu)

            prediction = tf.layers.dense(output, units=1, activation=tf.nn.sigmoid)
        return prediction


def multiplicative_attention(queries, keys, values, model_size=None):
    if model_size is None:
        model_size = tf.to_float(queries.get_shape().as_list()[-1])

    keys_T = tf.transpose(keys, [0, 2, 1])
    Q_K = tf.matmul(queries, keys_T) / tf.sqrt(model_size)
    attentions_weights = tf.nn.softmax(Q_K)
    multiplicative_att = tf.matmul(attentions_weights, values)
    return multiplicative_att, attentions_weights


def additive_attention(query, keys, values):
    raise NotImplementedError
