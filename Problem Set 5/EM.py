from math import pi
from math import e

def cluster_probablility(mean, std, point):
	return (1.0/(2*pi*std**2)**0.5) * e**(-1.0/(2*std**2) * abs(point - mean)**2)

mu1 = 6
mu2 = 7
std1 = 1
std2 = 2

tmp = []
points = [0, 1, 5, 6, 7]
p1_sum = 0
p2_sum = 0
for i, point in enumerate(points):
	p1 = cluster_probablility(mu1, std1, point)
	p2 = cluster_probablility(mu2, std2, point)
	p1_sum += p1
	p2_sum += p2
	tmp.append((p1, p2))
	# print 'Point', i, point, p2>p1, p1, p2 

cluster_probabilities = []
for x in tmp:
	cluster_probabilities.append((x[0]/sum(x), x[1]/sum(x)))

n1_hat = sum(x[0] for x in cluster_probabilities)
n2_hat = sum(x[1] for x in cluster_probabilities)


p1_hat = n1_hat / len(points)
p2_hat = n2_hat / len(points)

mu1_hat = (1.0/n1_hat) * sum([cluster_probabilities[i][0] * points[i] for i in range(len(points))])
mu2_hat = (1.0/n2_hat) * sum([cluster_probabilities[i][1] * points[i] for i in range(len(points))]) # sum([n[1] * x for n in cluster_probabilities for x in points])

std1_hat = (1.0/n1_hat) * sum([n[0] * (x - mu1_hat)**2 for n in cluster_probabilities for x in points])
std2_hat = (1.0/n2_hat) * sum([n[1] * (x - mu2_hat)**2 for n in cluster_probabilities for x in points])

for j in  cluster_probabilities:
	print j
print '\nn'
print n1_hat
print n2_hat, '\n'
print 'P'
print p1_hat
print p2_hat, '\n'
print 'mean'
print mu1_hat
print mu2_hat, '\n'
print 'std'
print std1_hat, std1_hat**0.5
print std2_hat, std2_hat**0.5