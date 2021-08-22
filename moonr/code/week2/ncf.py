import tensorflow as tf

print(tf.__version__)

data_path = 'file:///D:/project/电影推荐系统/data/data_suprise_format.csv'
file_path = tf.keras.utils.get_file("data_suprise_format.csv", data_path)


def get_dataset(file_path, label_name=None):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=16,
        label_name=label_name,
        na_value="?",
        num_epochs=5,
        ignore_errors=True)
    return dataset


raw_samples_data = get_dataset(file_path,'score')

test_dataset = raw_samples_data.take(1000)
train_dataset = raw_samples_data.skip(1000)

movie_col = tf.feature_column.categorical_column_with_identity(key='movieid', num_buckets=100000)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

user_col = tf.feature_column.categorical_column_with_identity(key='userid', num_buckets=60000)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)

inputs = {
    'movieid': tf.keras.layers.Input(name='movieid', shape=(), dtype='int32'),
    'userid': tf.keras.layers.Input(name='userid', shape=(), dtype='int32'),
}

movie_tower = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
user_tower = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)

interact_layer = tf.keras.layers.concatenate([movie_tower, user_tower])

for num_nodes in [128,128]:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)

output_layer = tf.keras.layers.Dense(1,activation='linear')(interact_layer)
neural_cf_model = tf.keras.Model(inputs, output_layer)

neural_cf_model.compile(loss='mse', optimizer='adam')

neural_cf_model.fit(train_dataset, epochs=10)
test_loss, test_accuracy = neural_cf_model.evaluate(test_dataset)
print('\n\nTest Loss {}'.format(test_loss))

