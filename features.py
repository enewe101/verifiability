from parc_reader import ParcCorenlpReader as P
import csv
from collections import defaultdict
from nltk.corpus import stopwords
import string
import pickle

#stopwords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn']
stopwords = ['and', 'the']
weasel_words_list = ['some', 'many', 'source', 'sources']
print stopwords
print string.punctuation
punct = list(string.punctuation)
print punct
import pickle


quoteMarkers = ['"', '``', "''"]
pluralPronouns = ['they', 'them', 'us', 'we']

def getAttr(attr):
	print attr
	filename = attr[0:8]
	folder = attr[4:6]

	parcFilePath = "/home/ndg/dataset/parc3/train/" + folder + '/' + filename + '.xml'
	corenlpPath =  "/home/ndg/dataset/ptb2-corenlp/CoreNLP/train/" + filename + '.xml'
	raw_text = "/home/ndg/dataset/ptb2-corenlp/masked_raw/train/" + filename

	parc_xml = open(parcFilePath).read()
	corenlp_xml = open(corenlpPath).read()
	raw_text = open(raw_text).read()

	article = P(corenlp_xml, parc_xml, raw_text)
	attribution = article.attributions[attr]
	return attribution, article



################### FEATURE EXTRACTION ###################
def featureExtract(attribution, article):

	featureDict = {}

	content = attribution['content']
	source = attribution['source']
	cue = attribution['cue']

	quoteDirectness = quoteType(content)
	featureDict['quoteType'] = quoteDirectness

	presenceEntitySource, typeEntitiesSource, pronoun, weasel_words = sourceFeats(source)
	featureDict['sourceEntityPresence'] = presenceEntitySource
	featureDict['typeEntities'] = typeEntitiesSource
	featureDict['pronounPresence'] = pronoun

	featureDict['weaselWordPresence'] = len(weasel_words) > 0
	featureDict['weaselWords'] = weasel_words
	


	if len(source) == 0:
		sourceHeadPlural = False
		sourceHeadLemma = []
		amods, dets = [], []
	else:
		sourceHead = article._find_head(source)
		sourceHeadPlural, sourceHeadLemma = headSource(sourceHead, article)
		amods, dets = sourceModifier(sourceHead)

	featureDict['sourcePlural'] = sourceHeadPlural
	featureDict['sourceLemma'] = sourceHeadLemma
	featureDict['amodPresence'] = len(amods) > 0
	featureDict['amods'] = amods
	featureDict['detPresence'] = len(dets) > 0
	featureDict['dets'] = dets

	contentBOW = bagOfWords(content)
	sourceBOW = bagOfWords(source)
	cueBOW = bagOfWords(cue)

	featureDict['contentWords'] = contentBOW
	featureDict['sourceWords'] = sourceBOW
	featureDict['cueBOW'] = cueBOW

	#for key in featureDict.keys():
	#	print key, featureDict[key]

	return featureDict

#page 99 pareti dissertation
def quoteType(content):
	typeAttr = ''
	words = [word['word'] for word in content]
	if words[0] in quoteMarkers and words[-1] in quoteMarkers:
		typeAttr = 'direct'
	elif len([word for word in words if word in quoteMarkers]) > 0:
		typeAttr = 'mixed'
	else:
		typeAttr = 'indirect'

	return typeAttr

def sourceFeats(source):
	presenceNamedEntity = False
	typeNamedEntity = []
	pronoun = False
	weasel_words = []
	for token in source:
		ne = token['ner']
		pos = token['pos']
		if ne != None:
			presenceNamedEntity = True
			typeNamedEntity.append(ne)
		if 'PRP' in pos:
			pronoun = True
		if token['lemma'] in weasel_words_list:
			weasel_words.append(token['lemma'])


	return presenceNamedEntity, list(set(typeNamedEntity)), pronoun, weasel_words

def headSource(sourceHead, article):
	plural = False
	lemma = []
	for head in sourceHead:
		if head['word'] == ',':
			continue
		if head['pos'].endswith('s'):
			plural = True
		elif head['lemma'] in pluralPronouns:
			plural = True

		if head['ner'] == 'ORGANIZATION':
			lemma.append('org')
		elif head['ner'] == 'PERSON':
			lemma.append('pers')
		elif head['ner'] == 'LOCATION':
			lemma.append('loc')
		else:
			lemma.append(head['lemma'])

	return plural, lemma

def sourceModifier(sourceHead):
	amods = []
	dets = []
	for sourceToken in sourceHead:
		if sourceToken['word'] == ',':
			continue
		children = sourceToken['children']
		for (relation, token) in children:
			if relation == 'amod':
				amods.append(token['lemma'])
			elif relation == 'det':
				dets.append(token['lemma'])

		print children

	return amods, dets

def bagOfWords(content):
	words = []
	for word in content:
		if word['ner'] == 'ORGANIZATION':
			words.append('org')
		elif word['ner'] == 'PERSON':
			words.append('pers')
		elif word['ner'] == 'LOCATION':
			words.append('loc')
		else:
			words.append(word['lemma'])

	filtered_words = [word for word in words if (word.lower() not in stopwords and word.lower() not in punct)]

	return filtered_words

################### VECTORIZE ###################

def vectorize(dataDict):
	featVector = []

	contentVocab = []
	sourceVocab = []
	cueVocab = []
	weaselWordsVocab = []
	typeEntitiesVocab = []
	sourceLemmaVocab = []
	amodVocab = []
	detsVocab = []

	ids = dataDict.keys()
	for attr in ids:
		features = dataDict[attr]
		contentVocab = contentVocab + features['contentWords']
		sourceVocab = sourceVocab + features['sourceWords']
		cueVocab = cueVocab + features['cueBOW']
		weaselWordsVocab = weaselWordsVocab + features['weaselWords']
		typeEntitiesVocab = typeEntitiesVocab + features['typeEntities']
		sourceLemmaVocab = sourceLemmaVocab + features['sourceLemma']
		amodVocab = amodVocab + features['amods']
		detsVocab = detsVocab +  features['dets']

	contentVocab = sorted(list(set(contentVocab)))
	sourceVocab = sorted(list(set(sourceVocab)))
	cueVocab = sorted(list(set(cueVocab)))
	weaselWordsVocab = sorted(list(set(weaselWordsVocab)))
	typeEntitiesVocab = sorted(list(set(typeEntitiesVocab)))
	sourceLemmaVocab = sorted(list(set(sourceLemmaVocab)))
	amodVocab = sorted(list(set(amodVocab)))
	detsVocab = sorted(list(set(detsVocab)))

	for attr in ids:
		thisVector = []
		features = dataDict[attr]
		if features['sourceEntityPresence'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		if features['sourcePlural'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		if features['weaselWordPresence'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		if features['amodPresence'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		if features['quoteType'] == 'direct':
			thisVector.append(0)
		elif features['quoteType'] == 'mixed':
			thisVector.append(1)
		else:
			thisVector.append(2)

		if features['pronounPresence'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		if features['detPresence'] == True:
			thisVector.append(1)
		else:
			thisVector.append(0)

		thisVector = thisVector + createListVectors(features['dets'], detsVocab)
		thisVector = thisVector + createListVectors(features['amods'], amodVocab)
		thisVector = thisVector + createListVectors(features['sourceLemma'], sourceLemmaVocab)
		thisVector = thisVector + createListVectors(features['typeEntities'], typeEntitiesVocab)
		thisVector = thisVector + createListVectors(features['weaselWords'], weaselWordsVocab)
		thisVector = thisVector + createListVectors(features['cueBOW'], cueVocab)
		thisVector = thisVector + createListVectors(features['sourceWords'], sourceVocab)
		thisVector = thisVector + createListVectors(features['contentWords'], contentVocab)

		thisVector = features['label'] + thisVector
		featVector.append(thisVector)

	return featVector

def createListVectors(thisList, compareList):
	indices = []
	lengthCompareList = len(compareList)
	zeroes = [0] * lengthCompareList
	if len(thisList) == 0:
		return zeroes
	for word in thisList:
		index = compareList.index(word)
		zeroes[index] = 1

	return zeroes




def execute(data):
	attr_ids = data.keys()
	i = 0
	
	'''
	for attr in attr_ids:
		attribution, article = getAttr(attr)
		featureDict = featureExtract(attribution, article)
		label = data[attr]
		featureDict['label'] = label
		data[attr] = featureDict
		
	

	pickle.dump(data, open('featureDict', 'wb'))
	'''
	
	data = pickle.load(open('featureDict', 'rb'))



	vectors = vectorize(data)

	pickle.dump(vectors, open('verifiabilityNumFeatures', 'wb'))



def openFile(pairwiseFile):
	finalData = defaultdict(list)


	with open(pairwiseFile, 'r') as f:
		reader = csv.reader(f)
		quotes = list(reader)

	for quote in quotes:
		string = quote[0]
		split = string.split('\t')
		attr_id = split[0]
		score = split[1]
		finalData[attr_id] = [score]

	return finalData


def main():
	pairwiseFile = "/home/ndg/project/shared_datasets/validators/100-pairwise-scores.tsv"
	data = openFile(pairwiseFile)
	execute(data)

if __name__ == '__main__':
   main()