# This Python file uses the following encoding: utf-8

from __future__ import division						
import pandas as pd
import nltk, re, math 
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.parse.stanford import StanfordDependencyParser

def preProcess(txt):

	table = dict([("aren't","are not"),("didn't","did not"),("doesn't","does not"),("don't","do not"),("won't","will not"),\
		("hasn't","has not"),("haven't","have not"),("isn't","is not"),("couldn't","could not"),("can't","can not"),\
		("wasn't","was not"),("weren't","were not"),("ă","a"),("ž","z"),("ó","o"),("ô","o"),("&quot;",""),("á","a")])

	for keyval in table.items():
		txt = txt.replace(keyval[0],keyval[1])
	return txt

stop_set = set(stopwords.words("english"))
def toWords(sent):										### also try without removing stop words
	sent = re.sub(r'[^\w\s]','',sent)
	word_only = word_tokenize(sent)
	words = [w for w in word_only if not w in stop_set]	# removes stop words
	#How do you remove duplicates from a list in whilst preserving order?  --> fastest
	seen = set()
	seen_add = seen.add
	return [x for x in words if not (x in seen or seen_add(x))]

def stemmingWords(sent):
	PS = PorterStemmer()
	stemmed_sent = []
	for w in sent:
		stemmed_sent.append(PS.stem(w))
	return stemmed_sent

def findSynonym(word):
	synonyms = []
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			synonyms.append(l.name())
	return synonyms	

def ngrams(sent,n):
	output = []
	for i in range(len(sent)-n+1):
		output.append(" ".join(sent[i:i+n]))
	return output

def skipGram(sent):
	output=[]
	n=3
	for i in range(len(sent)-n+1):
		skipGr = sent[i]+" "+sent[i+2]
		output.append(skipGr)
	return output

def findMatch(w1,w2):
	syn_w1 = findSynonym(w1)
	syn_w2 = findSynonym(w2)
	flag=0
	for x in syn_w1:
		for y in syn_w2:
			if x==y:
				return 1
	return 0

def textToVector(text):	
	WORD = re.compile(r'\w+')
	words = WORD.findall(text)
	return Counter(words)

def parseTree(sent):
	path_to_jar = '/home/knight/Downloads/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
	path_to_models_jar = '/home/knight/Downloads/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'
	dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

	result = dependency_parser.raw_parse(sent)
	depTree = result.next()
	return list(depTree.triples())

def LCS(X,Y):
    m = len(X)
    n = len(Y)
    L = [[None]*(n+1) for i in xrange(m+1)]			## 2d list of None*None
 
    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
    if n==0:
    	return n
    return L[m][n]/n

def feature1(txtw,hypw):
	unigram1 = ngrams(txtw,1)
	unigram2 = ngrams(hypw,1)
	inter = 0
	for w1 in txtw:
		for w2 in hypw:
			flag = findMatch(w1,w2)
			if flag==1:
				inter +=1
				break
	total = len(unigram2)
	if total==0:
		return 0
	return inter/total

def feature2(txtw,hypw):
	bigramTxt = ngrams(txtw,2)
	bigramHyp = ngrams(hypw,2)

	inter = 0
	for w1 in bigramTxt:
		for w2 in bigramHyp:
			if w1==w2:
				inter+=1
				break
	total = len(bigramHyp)
	if total==0:
		return total
	return inter/total

def feature3(txtw,hypw):
	return LCS(txtw,hypw)

def feature4(txtw,hypw):
	txtSkip = skipGram(txtw)
	hypSkip = skipGram(hypw)

	inter = 0
	for w1 in txtSkip:
		for w2 in hypSkip:
			if w1==w2:
				inter+=1
				break
	total = len(hypSkip)
	if total==0:
		return total
	return inter/total

def feature5(txtSt,hypSt):
	inter = 0
	for w1 in txtSt:
		for w2 in hypSt:
			if w1==w2:
				inter+=1
				break
	total = len(hypSt)
	if total==0:
		return total
	return inter/total

def feature6(txtw,hypw):			## LC substring
	
 	m = len(txtw)
 	n = len(hypw)
	LCSub = [[None]*(n+1) for i in xrange(m+1)]
	result = 0 
 
	for i in xrange(m+1):
		for j in xrange(n+1):
			if i==0 or j==0:
				LCSub[i][j] = 0
			elif txtw[i-1]==hypw[j-1]:
				LCSub[i][j] = LCSub[i-1][j-1]+1
				result = max(result,LCSub[i][j])
			else:
				LCSub[i][j] = 0
	if n==0:
		return n
	return result/n

def feature7(txt,hyp):		## cosine,  threshold >= 0.35
	
	vec1 = textToVector(txt)
	vec2 = textToVector(hyp)
	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

def feature8(s1,s2):		## jaccard similarity			## list parameters		## >=0.2
	set1 = set(s1)
	set2 = set(s2)
	inter = len(set1 & set2)
	union = len(set1 | set2)
	if union==0:
		return 0
	return inter/union

def feature9(posTxt,posHyp):
	nameEnt1 = []
	nameEnt2 = []
	for w in posTxt:
		if w[1]=='NNP' or w[1]=='NNPS':
			nameEnt1.append(w[0])
	for w in posHyp:
		if w[1]=='NNP' or w[1]=='NNPS':
			nameEnt2.append(w[0])
	set1 = set(nameEnt1)
	set2 = set(nameEnt2)
	inter = len(set1 & set2)
	n = len(set2)
	if n == 0:
		return n
	return inter/n

def compute(depRelation,rel1,rel2):
	toMatch = []
	for x in depRelation:
		if x[1]==rel1 or x[1]==rel2:
			toMatch.append(tuple((x[0][0],x[2][0])))
	return toMatch

def findDistance(word1,word2):							####improce here
	maxSimilar = 0.0
	syn1 = wordnet.synsets(word1)  
	syn2 = wordnet.synsets(word2)
	for w1 in syn1:
		for w2 in syn2:
			val = wordnet.wup_similarity(w1,w2)
			if val!=None:
				maxSimilar = max(maxSimilar,val)
	if maxSimilar<0.5:
		return 0
	return 0.5

def feature10(depRelation1,depRelation2):			#D3 
	toMatch = set()
	for x in depRelation1:
		if x[1]=='nsubj' or x[1]=='nsubjpass':
			toMatch.add(x[2][0])
	for x in depRelation2:
		if x[1]=='nsubj' or x[1]=='nsubjpass':
			if x[2][0] in toMatch:
				return 1
	return 0

################################## features for only NOUN, VERB, NUM to be added

def feature11(depRelation1,depRelation2):
	toMatch1 = compute(depRelation1,'nsubj','nsubjpass')
	toMatch2 = compute(depRelation2,'nsubj','nsubjpass')
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			flag=0
			if x[1]==y[1]:
				if x[0]==y[0]:
					flag = 1
				else:
					flag = findDistance(x[0],y[0])
			mx = max(mx,flag)
	return mx

def feature12(depRelation1,depRelation2):				## for max tuples 0

	toMatch1 = compute(depRelation1,'dobj','dobj')
	toMatch2 = compute(depRelation2,'dobj','dobj')
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			flag=0
			if x[1]==y[1]:
				if x[0]==y[0]:
					flag = 1
				else:
					flag = findDistance(x[0],y[0])
			mx = max(mx,flag)
	return mx

def calSimil(pos1,pos2):
	set1 = set(pos1)
	set2 = set(pos2)
	inter = len(set1 & set2)
	n = len(set2)
	if n == 0:
		return n
	return inter/n

def feature13(pos1,pos2):			## Noun
	noun1 = []
	noun2 = []
	for w in pos1:
		if w[1]=='NN' or w[1]=='NNS':
			noun1.append(w[0])
	for w in pos2:
		if w[1]=='NN' or w[1]=='NNS':
			noun2.append(w[0])
	return calSimil(pos1,pos2)

def feature14(pos1,pos2):				## verb 			## won't effect much
	match = {'VB','VBD','VBG','VBN','VBP','VBZ'}
	verb1 = []
	verb2 = []
	for w in pos1:
		if w[1] in match:
			verb1.append(w[0])
	for w in pos2:
		if w[1] in match:
			verb2.append(w[0])
	return calSimil(pos1,pos2)

def feature15(pos1,pos2):				## all
	return calSimil(pos1,pos2)

def feature16(depRelation1,depRelation2):										## all
	return calSimil(depRelation1,depRelation2)

def feature17(depRelation1,depRelation2):			## won't effect much
	toMatch1 = compute(depRelation1,'nummod','nummod')
	toMatch2 = compute(depRelation2,'nummod','nummod')
	#print(toMatch1,toMatch2)
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			flag = 0
			if x[1]==y[1]:
				if x[0]==y[0]:
					flag = 1
				else:
					flag = 0.5
			mx = max(flag,mx)
	return mx

def feature18(depRelation1,depRelation2):
	toMatch1 = compute(depRelation1,'det','det')
	toMatch2 = compute(depRelation2,'det','det')
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			flag = 0
			if x == y:
				flag = 1
			elif x[1]==y[1] or x[0]==y[0]:
				flag = 0.5
			mx = max(flag,mx)
	return mx

def feature19(depRelation1,depRelation2):				## done
	toMatch1 = compute(depRelation1,'compound','compound')
	toMatch2 = compute(depRelation2,'compound','compound')
	#print(toMatch1,toMatch2)
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			flag = 0
			if x[1]==y[1] and x[0]==y[0]:
				flag = 1
			elif x[1]==y[1] or x[0]==y[0]:
				flag = 0.5
			mx = max(flag,mx)
	return mx

def feature20(depRelation1,depRelation2):
	toMatch1 = compute(depRelation1,'case','case')
	toMatch2 = compute(depRelation2,'case','case')
	mx = 0
	for x in toMatch1:
		for y in toMatch2:
			if x[1]==y[1] and x[0]==y[0]:
				mx = 1
	return mx

def feature21(depRelation1,depRelation2):		## cross check
	s1 = compute(depRelation1,'nsubj','nsubjpass')
	s2 = compute(depRelation2,'nsubj','nsubjpass')
	o1 = compute(depRelation2,'dobj','dobj')
	o2 = compute(depRelation2,'dobj','dobj')
	
	for x in s1:
		for y in o2:
			if x==y:
				return 1
	for x in s2:
		for y in o1:
			if x==y:
				return 1
	return 0


def extractFeatures(entailment,txt,hyp):
	txtw = toWords(txt)
	hypw = toWords(hyp)

	txtSt = stemmingWords(txtw)
	hypSt = stemmingWords(hypw)

	posTxt = nltk.pos_tag(txtw)
	posHyp = nltk.pos_tag(hypw)

	depRelTxt = parseTree(txt)
	depRelHyp = parseTree(hyp)

	feat = []
	feat.append(feature1(txtw,hypw))
	feat.append(feature2(txtw,hypw))
	feat.append(feature3(txtw,hypw))
	feat.append(feature4(txtw,hypw))
	feat.append(feature5(txtSt,hypSt))
	feat.append(feature6(txtw,hypw))

	feat.append(feature7(txt,hyp))
	feat.append(feature8(txtSt,hypSt))
	feat.append(feature9(posTxt,posHyp))

	feat.append(feature10(depRelTxt,depRelHyp))
	feat.append(feature11(depRelTxt,depRelHyp))
	feat.append(feature12(depRelTxt,depRelHyp))

	feat.append(feature13(posTxt,posHyp))
	feat.append(feature14(posTxt,posHyp))
	feat.append(feature15(posTxt,posHyp))

	feat.append(feature16(depRelTxt,depRelHyp))
	feat.append(feature17(depRelTxt,depRelHyp))
	feat.append(feature18(depRelTxt,depRelHyp))
	feat.append(feature19(depRelTxt,depRelHyp))
	feat.append(feature20(depRelTxt,depRelHyp))
	feat.append(feature21(depRelTxt,depRelHyp))

	if entailment =='YES':
		entailment = 1
	else:
		entailment = 0

	feat.append(entailment)
	return feat

########################  Modify the main function accordingly  ######################################

def mainFunction():

	tree = ET.parse('RTE3_pairs_dev-set-final.xml')				## will read directly
	root = tree.getroot()
	#data_size = root.shape[0]				## need to confirm size
	
	allFeat = []
	for c in xrange(0,800):		
		ids = root[c].attrib['id']
		entailment = root[c].attrib['entailment']
		task = root[c].attrib['task']
		txt = root[c][0].text
		hyp = root[c][1].text
	
		#txt = preProcess(txt)
		#hyp = preProcess(hyp)

		#if c%10==0:
		#	print(ids,hyp)
		print(c,hyp)
		allFeat.append(extractFeatures(entailment,txt,hyp))
	df = pd.DataFrame(allFeat,columns=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11',\
		'F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','Entailment'])
	with open('trainRTE3.csv', 'a') as file:
		#df.to_csv(file,index=False)
		df.to_csv(file,index=False,header=False)



mainFunction()



