import numpy as np
import project1_code as p1

######################
# INITIALIZE
######################
adjectives = p1.extract_set('adjectives.txt')
dictionary = p1.extract_dictionary('train-tweet.txt')
labels = p1.read_vector_file('train-answer.txt')
feature_matrix = p1.extract_feature_vectors_with_keywords('train-tweet.txt', dictionary, adjectives)
######################
# AVERAGER
######################
average_theta, average_theta_0 = p1.averager(feature_matrix, labels)
label_output = p1.perceptron_classify(feature_matrix, average_theta_0, average_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Averager gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")

######################
# PERCEPTRON
######################
perceptron_theta, perceptron_theta_0 = p1.train_perceptron(feature_matrix, labels)
label_output = p1.perceptron_classify(feature_matrix, perceptron_theta_0, perceptron_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Perceptron gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")

######################
# PASSIVE-AGRESSIVE
######################
pa_theta, pa_theta_0 = p1.train_passive_agressive(feature_matrix, labels, 50)
label_output = p1.perceptron_classify(feature_matrix, pa_theta_0, pa_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Passive-agressive gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")

# ######################
# # CROSS VALIDATION
# ######################
K = 40
print 'K =', K
averager_performance = p1.cross_validation(feature_matrix, labels, K, 0)
print("Averager gets " + str(averager_performance) + "% correct on average via cross-validation.")
perceptron_performance = p1.cross_validation(feature_matrix, labels, K, 1)
print("Perceptron gets " + str(perceptron_performance) + "% correct on average via cross-validation.")
pa_performance = p1.cross_validation(feature_matrix, labels, K, 2, T = 100)
print("Passive-agressive gets " + str(pa_performance) + "% correct on average via cross-validation.")
