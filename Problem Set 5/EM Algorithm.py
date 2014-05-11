import numpy as np
from math import pi
from math import e

def cluster_probablility(mean, std, point):
	return (1.0/(2*pi*std**2)**0.5) * e**(-1.0/(2*std**2) * abs(point - mean)**2)

def EStep(ps, means, stds, points):
	pits = np.zeros([len(ps), len(points)])
	for i, point in enumerate(points):
		p = 0
		for j, cluster in enumerate(ps):
			a = ps[j] * cluster_probablility(means[j], stds[j], point)
			pits[j][i] = a
			p += a
		pits.T[i] /= p
	return pits

def MStep(pits, means, stds, points):
	# print 'pits', pits
	pits = np.array(pits)
	nH = np.sum(pits, axis = 1)
	nH = nH.astype(float)	
	# print 'NH', nH
	pH = nH / len(points)
	meanH = np.zeros([len(pits)])
	for i, p in enumerate(pits):
		for j, point in enumerate(points):
			meanH[i] += pits[i][j] * points[j]
		meanH[i] /= nH[i]
	stdH = [None] * len(pits)
	for i, p in enumerate(pits):
		var_sum = 0
		for j, point in enumerate(points):
			var_sum += pits[i][j] * abs(points[j] - means[i])**2
		stdH[i] = var_sum / nH[i]
	return pH, meanH, stdH**0.5

points = [0, 1, 5, 6, 7]

ps = [0.5, 0.5]
means = [6, 7]
stds = [1, 2]




print '\n\n'

res = EStep(ps, means, stds, points)
for i in range(len(res[0])):
	print res[0][i], res[1][i], res[1][i] > res[0][i]

# pits = EStep(ps, means, stds, points)
# p, mean, std = MStep(pits, means, stds, points)
# count = 1
# pOld = np.array([0])
# while (count < 100 and p.all() != pOld.all()):
# 	# print count
# 	pits = EStep(p, mean, std, points)
# 	pOld = p
# 	p, mean, std = MStep(pits, means, stds, points)
# 	count += 1
# print count

# print '\n', mean, '\n'
# print std, '\n'
# print [std[0]**0.5, std[1]**0.5], '\n'