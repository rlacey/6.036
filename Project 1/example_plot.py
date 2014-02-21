import numpy as np
import project1_code as p1

# pos1 = [-0.3, 0.4]
# pos2 = [0.2, 0.3]
# neg1 = [-0.1, -0.1]
# neg2 = [0.3, 0.1]

# feature_matrix = [pos1, pos2, neg1, neg2]
# labels = [1, 1, -1, -1]

# p1.plot_2d_examples(feature_matrix, labels, 0, [0.25, 0.6])

loose_points = np.array([
                [-3,4],
                [-2,3],
                [2,4],
                [4,2],
                [-3,-2],
                [0,-2],
                [3,-3]])

loose_labels = np.array([1,1,1,1,-1,-1,-1])

average_theta, average_theta_0 = p1.averager(loose_points, loose_labels)
p1.plot_2d_examples(loose_points, loose_labels, average_theta_0, average_theta, 'Averager - loose points')

perceptron_theta, perceptron_theta_0 = p1.train_perceptron(loose_points, loose_labels)
p1.plot_2d_examples(loose_points, loose_labels, perceptron_theta_0, perceptron_theta, 'Perceptron - loose points')

pa_theta, pa_theta_0 = p1.train_passive_agressive(loose_points, loose_labels, 1000)
p1.plot_2d_examples(loose_points, loose_labels, pa_theta_0, pa_theta, 'Passive Agressive - loose points')

close_points = np.array([
                [-1,-1.25],
                [-1.5, -1],
                [1,4],
                [1.5,1.5],
                [4,10],
                [-1,-1]])

close_labels = np.array([-1, 1, 1, -1, 1, -1])

average_theta, average_theta_0 = p1.averager(close_points, close_labels)
p1.plot_2d_examples(close_points, close_labels, average_theta_0, average_theta, 'Averager - close points')

perceptron_theta, perceptron_theta_0 = p1.train_perceptron(close_points, close_labels)
p1.plot_2d_examples(close_points, close_labels, perceptron_theta_0, perceptron_theta, 'Perceptron - close points')

pa_theta, pa_theta_0 = p1.train_passive_agressive(close_points, close_labels, 1000)
p1.plot_2d_examples(close_points, close_labels, pa_theta_0, pa_theta, 'Passive Agressive - close points')
