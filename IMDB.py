(train_data, train_labels), (test_data, test_labels)=imdb.load_data(path="D:/cv/DataSets/imdb/imdb.npz", num_words=10000)
word_index=imdb.get_word_index(path="D:/cv/DataSets/imdb/imdb_word_index.json")
# # train_data, train_labels = f['x_train'], f['y_train']
# # test_data, test_labels = f['x_test'], f['y_test']
# (train_data, train_labels), (test_data, test_labels) = path.load_data(num_words=10000)
