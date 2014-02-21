import numpy as np
import project1_code as p1

######################
# INITIALIZE
######################
adjectives = p1.extract_set('adjectives.txt')
dictionary = p1.extract_dictionary('train-tweet.txt')
train_labels = p1.read_vector_file('train-answer.txt')
train_feature_matrix = p1.extract_feature_vectors_with_keywords('train-tweet.txt', dictionary, adjectives)
test_feature_matrix = p1.extract_feature_vectors_with_keywords('test-tweet.txt', dictionary, adjectives)

######################
# TRAIN
######################
pa_theta, pa_theta_0 = p1.train_passive_agressive(train_feature_matrix, train_labels, 1000)

######################
# CLASSIFY
######################
label_output = p1.perceptron_classify(test_feature_matrix, pa_theta_0, pa_theta)

print train_feature_matrix.shape
print test_feature_matrix.shape
p1.write_label_answer(label_output, 'tweet_labels.txt')
