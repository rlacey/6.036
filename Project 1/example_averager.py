import numpy as np
import project1_code as p1
    
dictionary = p1.extract_dictionary('M:/School/6.036/Project 1/train-tweet.txt')
labels = p1.read_vector_file('M:/School/6.036/Project 1/train-answer.txt')
feature_matrix = p1.extract_feature_vectors('M:/School/6.036/Project 1/train-tweet.txt', dictionary)

average_theta = p1.averager(feature_matrix, labels)
label_output = p1.perceptron_classify(feature_matrix, 0, average_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Averager gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")

perceptron_theta, perceptron_theta_0 = p1.train_perceptron(feature_matrix, labels)
label_output = p1.perceptron_classify(feature_matrix, perceptron_theta_0, perceptron_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Perceptron gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")
                
perceptron_performance = p1.cross_validation(feature_matrix, labels, 5)
print("Perceptron gets " + str(perceptron_performance) + "% correct on average via cross-validation.")