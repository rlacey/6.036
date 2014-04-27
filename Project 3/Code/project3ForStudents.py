import numpy as np
import math
import re

def loadTrainData():
	# load the training corpus

	# load vocabulary and assign index
	vocmap = dict()
	f = open('freqwords', 'r')
	c = 0
	for line in f:
		vocmap[line.rstrip()] = c
		c += 1
	vocmap['UNKA'] = c
	f.close()
	print 'Vocabulary size:', len(vocmap)

	# load sentence and label
	f = open('wsj.0-18', 'r')
	wordList = []
	labelList = []
	for line in f:
		data = line.strip().split()
		tmp = data[0::2]
		wordList.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
		tmp = data[1::2]
		labelList.append(tmp)
	f.close()

	# construct tagset and assign index
	tagmap = dict()
	for sent in labelList:
		for i in range(len(sent)):
			if not sent[i] in tagmap:
				tagmap[sent[i]] = len(tagmap)
			sent[i] = tagmap[sent[i]]
	print 'Tagset size:', len(tagmap)

	return wordList, labelList, vocmap, tagmap

def splitDataAndGetCounts(ratio, wordList, labelList, vocmap, tagmap):
	#split the fully labeled data by the ratio and return the count and all sentences

	# split data
	labelNum = int(ratio * len(wordList))
	unlabelWordList = wordList[labelNum:]
	unlabelLabelList = labelList[labelNum:]
	wordList = wordList[:labelNum]
	labelList = labelList[:labelNum]

	# construct parameter table
	W = len(vocmap)
	T = len(tagmap)

	t = np.zeros(T)
	tw = np.zeros((T, W))
	tpw = np.zeros((T, W))
	tnw = np.zeros((T, W))

	# calculate count to the table
	for i in range(len(wordList)):
		word = wordList[i]
		label = labelList[i]
		for j in range(len(word)):
			t[label[j]] += 1.0
			tw[label[j], word[j]] += 1.0
			if j > 0:
				tpw[label[j], word[j-1]] += 1.0
			if j < len(word) - 1:
				tnw[label[j], word[j+1]] += 1.0

	# smoothing
	smoothing(1.0, t, tw, tpw, tnw)

	return t, tw, tpw, tnw, unlabelWordList, unlabelLabelList

def smoothing(alpha, t, tw, tpw, tnw):
	# adding the smooth counts to the original ones
	T, W = tw.shape
	t += alpha / T
	tw += alpha / (T * W)
	tpw += alpha / (T * W)
	tnw += alpha / (T * W)

def loadTestData(vocmap, tagmap):
	# load and return the test data and gold label, converted into index

	# a list of sentences, each element contains a list of words
	wordList = []
	# a list of sentences, each element contains a list of labels (POS)
	labelList = []

	f = open('wsj.19-21', 'r')
	for line in f:
		data = line.strip().split()
		tmp = data[0::2]
		wordList.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
		tmp = data[1::2]
		labelList.append([tagmap[s] for s in tmp])
	f.close()

	return wordList, labelList

def Mstep(et, etw, etpw, etnw, t, tw, tpw, tnw):
	# ratio: the split ratio of labeled and unlabled data; used to compute weight of real counts

	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)

	# t: real counts for \theta(t)
	# tw: real counts for \theta_0(w|t)
	# tpw: real counts for \theta_{-1}(w|t)
	# tnw: real counts for \theta_{+1}(w|t)

	# pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)

	T, W = etw.shape
	pt = np.zeros(et.shape)
	ptw = np.zeros(etw.shape)
	ptpw = np.zeros(etpw.shape)
	ptnw = np.zeros(etnw.shape)

	# c is the weight of real count
	c = 100.0

	# Estimate parameters pt, ptw, ptpw, ptnw based on the expected counts and real counts
	# Your code here:
	nt   = c * t   + et
	ntw  = c * tw  + etw
	ntpw = c * tpw + etpw
	ntnw = c * tnw + etnw

	pt = nt / np.sum(nt)
	ptw = ntw / np.array([np.sum(ntw, axis = 1)]).T
	ptpw = ntpw / np.array([np.sum(ntpw, axis = 1)]).T
	ptnw = ntnw / np.array([np.sum(ntnw, axis = 1)]).T

	return pt, ptw, ptpw, ptnw

def EstepA(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))

	for sent in wordList:
		for pos in range(len(sent)):

			# Compute the posterior for each word
			p = pt * ptw[:, sent[pos]]
			p /= np.sum(p)

			# Accumulate expected counts based on posterior
			et += p
			etw[:, sent[pos]] += p

	return et, etw, etpw, etnw

def likelihoodA(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model A
	l = sum([sum([np.log(sum(pt * ptw[:, word])) for word in sent]) for sent in wordList])
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)))
	return l

def EstepB(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))

	for sent in wordList:
		for pos in range(len(sent)):

			# Compute the posterior for each word
			p = pt * ptw[:, sent[pos]]
			if pos > 0:
				p *= ptpw[:, sent[pos-1]]
			p /= np.sum(p)

			# Accumulate expected counts based on posterior
			et += p
			etw[:, sent[pos]] += p
			if pos > 0:
				etpw[:, sent[pos-1]] += p

	return et, etw, etpw, etnw

def likelihoodB(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model B
	l = sum([sum([np.log(sum(pt * ptw[:, sent[i]] * ptpw[:, sent[i-1]])) if i > 0 else np.log(sum(pt * ptw[:, sent[i]])) for i in range(len(sent))]) for sent in wordList])
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)))
	return l

def EstepC(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))

	for sent in wordList:
		for pos in range(len(sent)):
			# Compute the posterior for each word
			p = pt * ptw[:, sent[pos]]
			if pos > 0:
				p *= ptpw[:, sent[pos-1]]
			if pos < len(sent)-1:
				p *= ptnw[:, sent[pos+1]]
			p /= np.sum(p)

			# Accumulate expected counts based on posterior
			et += p
			etw[:, sent[pos]] += p
			if pos > 0:
				etpw[:, sent[pos-1]] += p
			if pos < len(sent)-1:
				etnw[:, sent[pos+1]] += p

	return et, etw, etpw, etnw

def likelihoodC(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model C
	l = 0.0
	for sent in wordList:
		for pos in range(len(sent)):
			prob = pt * ptw[:, sent[pos]]
			if pos > 0:
				prob = prob* ptpw[:, sent[pos-1]]
			if pos < len(sent) - 1:
				prob = prob * ptnw[:, sent[pos+1]]
			l += np.log(sum(prob))
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)) + np.sum(tnw * np.log(ptnw)))
	return l

def predictA(wordList, pt, ptw, ptpw, ptnw, tags, voc):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)
	best  = (100, None, None)
	mahTags = dict (zip(tags.values(),tags.keys()))
	wurds = dict (zip(voc.values(),voc.keys()))
	print len(mahTags), mahTags
	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	pred = []

	# Predict tag index in each sentence based on Model A
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):
			# Your code here:
			pred_tag = (pt * ptw[:,sent[pos]]).argmax(axis=0)
			maxVal = np.amax(pt * ptw[:,sent[pos]], axis=0) / np.sum((pt * ptw[:,sent[pos]]))
			if maxVal < best[0] and re.match('^[\w-]+$', mahTags[pred_tag]) is not None and re.match('^[\w-]+$', wurds[sent[pos]]) is not None:
				best = (maxVal, mahTags[pred_tag], wurds[sent[pos]])
			# append the prediction to the list
			cur_pred.append(pred_tag)
		pred.append(cur_pred)
	print 'Best', best
	return pred

def predictB(wordList, pt, ptw, ptpw, ptnw, tags, voc):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)


	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	pred = []

	# Predict tag index in each sentence based on Model B
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):

			# Compute the posterior for each word
			prediction = pt * ptw[:, sent[pos]]
			if pos > 0:
				prediction *= ptpw[:, sent[pos-1]]

			pred_tag = prediction.argmax(axis=0)

			cur_pred.append(pred_tag)
		pred.append(cur_pred)

	return pred

def predictC(wordList, pt, ptw, ptpw, ptnw, tags, voc):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)

	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	pred = []

	# Predict tag index in each sentence based on Model C
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):
			prediction = pt * ptw[:, sent[pos]]
			if pos > 0:
				prediction *= ptpw[:, sent[pos-1]]
			if pos < len(sent) - 1:
				prediction *= ptnw[:, sent[pos+1]]

			pred_tag = prediction.argmax(axis=0)

			# append the prediction to the list
			cur_pred.append(pred_tag)
		pred.append(cur_pred)

	return pred

def evaluate(labelList, pred):
	# compute accuracy
	if len(labelList) != len(pred):
		print 'number of sentences mismatch!'
		return None

	acc = 0.0
	total = 0.0
	for i in range(len(labelList)):
		if len(labelList[i]) != len(pred[i]):
			print 'length mismatch on sentence', i
			return None
		total += len(labelList[i])
		acc += sum([1 if labelList[i][j] == pred[i][j] else 0 for j in range(len(labelList[i]))])
	return acc / total

def task1():
	# Hint: This function is fully implemented. Just call it and report your result

	# Test each model given labeled data
	# load the count from training corpus
	wordList, labelList, vocmap, tagmap = loadTrainData()
	t, tw, tpw, tnw, unlabelWordList, unlabelLabelList = splitDataAndGetCounts(1.0, wordList, labelList, vocmap, tagmap)
	# estimate the parameters
	pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), t, tw, tpw, tnw)
	# load the testing data
	wordList, labelList = loadTestData(vocmap, tagmap)

	# predict using each model and evaluate
	pred = predictA(wordList, pt, ptw, ptpw, ptnw)
	print "Model A accuracy:", evaluate(labelList, pred)

	pred = predictB(wordList, pt, ptw, ptpw, ptnw)
	print "Model B accuracy:", evaluate(labelList, pred)

	pred = predictC(wordList, pt, ptw, ptpw, ptnw)
	print "Model C accuracy:", evaluate(labelList, pred)

def task2():
	# Hint: This function is fully implemented. Just call it and report your result
	# You will get (1) the accuracy trained only on the labeled data, (2) the log-likelihood and model accuracy after each iteration

	taskem(0.5)

def task3():
	# Hint: This function is fully implemented. Just call it and report your result
	# You will get (1) the accuracy trained only on the labeled data, (2) the log-likelihood and model accuracy after each iteration

	taskem(0.01)

def taskem(ratio):
	wordList, labelList, vocmap, tagmap = loadTrainData()

	print ratio, 'labeled,', 1-ratio, 'unlabeled:'

	t, tw, tpw, tnw, unlabelWordList, unlabelLabelList = splitDataAndGetCounts(ratio, wordList, labelList, vocmap, tagmap)

	# try different models
	estepFunc = [EstepA, EstepB, EstepC]
	likelihoodFunc = [likelihoodA, likelihoodB, likelihoodC]
	predictFunc = [predictA, predictB, predictC]
	name = ['A', 'B', 'C']

	for m in range(len(name)):
		print 'Use model ' + name[m] + ':'
		# estimate on labeled data only
		pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), t, tw, tpw, tnw)
		pred = predictFunc[m](unlabelWordList, pt, ptw, ptpw, ptnw, tagmap, vocmap)
		print "Model accuracy on labeled data:", evaluate(unlabelLabelList, pred)

		# use the uniform distribution as initialization
		pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), np.ones(t.shape), np.ones(tw.shape), np.ones(tpw.shape), np.ones(tnw.shape))
		# run EM
		maxIter = 4
		Estep = estepFunc[m]
		likelihood = likelihoodFunc[m]
		for iter in range(maxIter):
			et, etw, etpw, etnw = Estep(pt, ptw, ptpw, ptnw, unlabelWordList)
			pt, ptw, ptpw, ptnw = Mstep(et, etw, etpw, etnw, t, tw, tpw, tnw)
			l = likelihood(pt, ptw, ptpw, ptnw, unlabelWordList, t, tw, tpw, tnw)
			pred = predictFunc[m](unlabelWordList, pt, ptw, ptpw, ptnw, tagmap, vocmap)
			print 'Iter', iter + 1, 'Log-likelihood =', l, "Model accuracy:", evaluate(unlabelLabelList, pred)

# print 'Task 1'
# task1()
print 'Task 2'
task2()
print 'Task 3'
task3()

