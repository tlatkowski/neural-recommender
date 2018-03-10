import tensorflow as tf


class MatrixFactorization:

    def __init__(self, num_users, num_items, user_embedding_size, item_embedding_size):

        with tf.Graph().as_default() as graph:

            self.users = tf.placeholder(dtype=tf.float32, shape=[None])
            self.items = tf.placeholder(dtype=tf.float32, shape=[None])

            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None])

            with tf.name_scope('embeddings'):
                users_embeddings = tf.get_variable('users_embeddings', [num_users, user_embedding_size])
                items_embeddings = tf.get_variable('items_embeddings', [num_items, item_embedding_size])

                users_embedded = tf.gather(users_embeddings, self.users)
                items_embedded = tf.gather(items_embeddings, self.items)

            with tf.name_scope('factorization'):
                users_items = tf.multiply(users_embedded, items_embedded)

                prediction = tf.layers.dense(users_items, units=1, activation=tf.nn.sigmoid)
