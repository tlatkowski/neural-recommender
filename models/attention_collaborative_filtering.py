import tensorflow as tf


class AttentionCollaborativeFiltering:

    def __init__(self, num_users, num_items, user_embedding_size, item_embedding_size, learning_rate=0.001, layers_sizes=[200, 100]):

        with tf.Graph().as_default() as graph:
            self.users = tf.placeholder(dtype=tf.int32, shape=[None])
            self.items = tf.placeholder(dtype=tf.int32, shape=[None])

            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None])

            with tf.name_scope('embeddings'):
                users_embeddings = tf.get_variable('users_embeddings', [num_users, user_embedding_size])
                items_embeddings = tf.get_variable('items_embeddings', [num_items, item_embedding_size])

                users_embedded = tf.gather(users_embeddings, self.users)
                items_embedded = tf.gather(items_embeddings, self.items)

            with tf.name_scope('factorization'):
                users_self_attention, _ = multiplicative_attention(users_embedded, users_embedded, users_embedded)
                items_self_attention, _ = multiplicative_attention(items_embedded, items_embedded, items_embedded)
                users_items_attention, _ = multiplicative_attention(users_embedded, items_embedded,
                                                                    items_embedded)  # TODO check correctness
                users_items = tf.concat([users_self_attention, items_self_attention, users_items_attention], axis=0)

                output = users_items
                for i, layer_size in enumerate(layers_sizes):
                    output = tf.layers.dense(output, units=layers_sizes[i], activation=tf.nn.relu)

                prediction = tf.layers.dense(output, units=1, activation=tf.nn.sigmoid)

            with tf.name_scope('loss'):
                self.loss = tf.losses.sigmoid_cross_entropy(self.ratings, prediction)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


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
