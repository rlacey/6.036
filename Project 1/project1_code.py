from string import punctuation, digits
import numpy as np
import matplotlib.pyplot as plt

def extract_words(input_string):
    """
      Returns a list of lowercase words in a strong.
      Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def extract_set(file):
    """
      Given a text file, returns a dictionary of unique words.
      Each line is passed into extract_words, and a list on unique
      words is maintained.
    """
    s = set([])

    f = open(file, 'r')
    for line in f:
        flist = extract_words(line)

        for word in flist:
            if(word not in s):
                s.add(word)

    f.close()

    return s

def extract_dictionary(file):
    """
      Given a text file, returns a dictionary of unique words.
      Each line is passed into extract_words, and a list on unique
      words is maintained.
    """
    dict = []

    f = open(file, 'r')
    for line in f:
        flist = extract_words(line)

        for word in flist:
            if(word not in dict):
                dict.append(word)

    f.close()

    return dict

def extract_feature_vectors(file, dict):
    """
      Returns a bag-of-words representation of a text file, given a dictionary.
      The returned matrix is of shape (m, n), where the text file has m non-blank
      lines, and the dictionary has n entries.
    """
    f = open(file, 'r')
    num_lines = 0

    for line in f:
        if(line.strip()):
            num_lines = num_lines + 1

    f.close()

    feature_matrix = np.zeros([num_lines, len(dict)])

    f = open(file, 'r')
    pos = 0

    for line in f:
        if(line.strip()):
            flist = extract_words(line)
            for word in flist:
                if(word in dict):
                    feature_matrix[pos, dict.index(word)] = 1
            pos = pos + 1

    f.close()

    return feature_matrix

def extract_feature_vectors_with_keywords(file, dict, keys):
    """
      Returns a bag-of-words representation of a text file, given a dictionary.
      The returned matrix is of shape (m, n), where the text file has m non-blank
      lines, and the dictionary has n entries.
    """
    f = open(file, 'r')
    num_lines = 0

    for line in f:
        if(line.strip()):
            num_lines = num_lines + 1

    f.close()

    feature_matrix = np.zeros([num_lines, len(dict)])

    f = open(file, 'r')
    pos = 0

    for line in f:
        if(line.strip()):
            flist = extract_words(line)
            for word in flist:
                if(word in dict):
                    if word in keys:
                        feature_matrix[pos, dict.index(word)] = 2
                    else:
                        feature_matrix[pos, dict.index(word)] = 1
            pos = pos + 1

    f.close()

    return feature_matrix

def averager(feature_matrix, labels):
    """
      Implements a very simple classifier that averages the feature vectors multiplied by the labels.
      Inputs are an (m, n) matrix (m data points and n features) and a length m label vector.
      Returns a length-n theta vector (theta_0 is 0).
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_0 = 0

    for i in xrange(0, nsamples):
        label = labels[i]
        sample_vector = feature_matrix[i, :]
        theta_vector = theta_vector + label*sample_vector
        theta_0 = theta_0 + label
    return theta_vector, theta_0

def train_perceptron(feature_matrix, labels):
    """
      Implements the perceptron classifier penalizes misclassified points.
      Inputs are an (m, n) matrix (m data points and n features) and a length m label vector.
      Returns a length-n theta vector and offset.
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_0 = 0

    counter = 0
    misclassification = True

    while(misclassification and counter < 5000):
        misclassification = False
        counter += 1
        for i in range(0, nsamples):
            label = labels[i]
            feature_vector = feature_matrix[i, :]
            perceptron_output = np.dot(theta_vector, feature_vector) + theta_0
            if (label * perceptron_output <= 0):
                theta_vector = theta_vector + label * feature_vector
                theta_0 = theta_0 + label
                misclassification = True
    return theta_vector, theta_0

def train_passive_agressive(feature_matrix, labels, T):
    (nsamples, nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_0 = 0

    for t in xrange(T):
        for i in range(0, nsamples):
            label = labels[i]
            feature_vector = feature_matrix[i, :]
            loss = 1 - label * (np.dot(theta_vector, feature_vector) + theta_0) if label * (np.dot(theta_vector, feature_vector) + theta_0) <= 1 else 0
            mag  = feature_vector.dot(feature_vector) + 1
            alpha = loss / mag
            theta_vector = theta_vector + alpha * label * feature_vector
            theta_0 = theta_0 + alpha * label
    return theta_vector, theta_0

def read_vector_file(fname):
    """
      Reads and returns a vector from a file.
    """
    return np.genfromtxt(fname)

def perceptron_classify(feature_matrix, theta_0, theta_vector):
    """
      Classifies a set of data points given a weight vector and offset.
      Inputs are an (m, n) matrix of input vectors (m data points and n features),
      a real number offset, and a length n parameter vector.
      Returns a length m label vector.
    """
    (nsamples, nfeatures) = feature_matrix.shape
    label_output = np.zeros([nsamples])

    for i in xrange(0, nsamples):
        sample_features = feature_matrix[i, :]
        perceptron_output = theta_0 + np.dot(theta_vector, sample_features)

        if(perceptron_output > 0):
            label_output[i] = 1
        else:
            label_output[i] = -1

    return label_output

def cross_validation(feature_matrix, labels, k, flag = 0, T = 100):
    """
        Runs k-fold cross validation on feature matrix.
        Flag  0 : averager
              1 : perceptron
              2 : passive agressive
        Returns average percentage of correctly labeled points.
    """
    (nsamples, nfeatures) = feature_matrix.shape
    nlabels = len(labels)
    subset_size = nsamples / k
    percentages = []

    for i in xrange(k):
        cross_feature_matrix = np.concatenate((feature_matrix[:i*subset_size], feature_matrix[(i+1)*subset_size:]))
        cross_labels = np.concatenate((labels[:i*subset_size], labels[(i+1)*subset_size:]))
        test_matrix = feature_matrix[i*subset_size:(i+1)*subset_size]
        test_labels = labels[i*subset_size:(i+1)*subset_size]

        if flag == 0:
            theta_vector, theta_0 = averager(cross_feature_matrix, cross_labels)
        elif flag == 1:
            theta_vector, theta_0 = train_perceptron(cross_feature_matrix, cross_labels)
        elif flag == 2:
            theta_vector, theta_0 = train_passive_agressive(cross_feature_matrix, cross_labels, T)

        label_output = perceptron_classify(test_matrix, theta_0, theta_vector)

        correct = 0
        for i in xrange(0, len(label_output)):
            if(label_output[i] == test_labels[i]):
                correct = correct + 1

        percentages.append(100.0 * correct / len(label_output))

    return sum(percentages) / len(percentages)


def write_label_answer(vec, outfile):
    """
      Outputs your label vector the a given file.
      The vector must be of shape (70, ) or (70, 1),
      i.e., 70 rows, or 70 rows and 1 column.
    """

    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    for v in vec:
        if((v != -1.0) and (v != 1.0)):
            print("Invalid value in input vector.")
            print("Aborting write.")
            return

    np.savetxt(outfile, vec)


def plot_2d_examples(feature_matrix, labels, theta_0, theta, title = None):
    """
      Uses Matplotlib to plot a set of labeled instances, and
      a decision boundary line.
      Inputs: an (m, 2) feature_matrix (m data points each with
      2 features), a length-m label vector, and hyper-plane
      parameters theta_0 and length-2 vector theta.
    """

    cols = []
    xs = []
    ys = []

    for i in xrange(0, len(labels)):
        if(labels[i] == 1):
            cols.append('b')
        else:
            cols.append('r')
        xs.append(feature_matrix[i][0])
        ys.append(feature_matrix[i][1])

    plt.scatter(xs, ys, s=40, c=cols)

    [xmin, xmax, ymin, ymax] = plt.axis()

    linex = []
    liney = []
    for x in np.linspace(xmin, xmax):
        linex.append(x)
        if(theta[1] != 0.0):
            y = (-theta_0 - theta[0]*x) / (theta[1])
            liney.append(y)
        else:
            liney.append(0)

    plt.plot(linex, liney, 'k-')

    if title:
        plt.title(title)

    plt.show()
