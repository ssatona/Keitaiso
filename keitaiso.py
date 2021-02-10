# -*- coding:utf-8 -*-
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import re
import pandas as pd
import sqlite3
import gensim
from gensim import corpora, models
from collections import defaultdict
import codecs
import pickle
import nltk
#nltk.download()
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
#from pattern import pluralize, singularize
from nltk.probability import FreqDist
#nltk.download('stopwords')
#nltk.download('universal_tagset')
import inflect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('PorterStemmer')
#nltk.download('wordnet_ic')

def nofilter_exceptStopWords(sentence):

	#csv_input = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
	#stopwords=csv_input['stopwords'].values.tolist()
	excel_moji = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
	num = re.compile("([0-9０-９])")
	kana = re.compile('([｡ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊｲﾌﾍﾎﾏﾐﾑﾒﾓﾗﾘﾙﾚﾛﾔﾕﾖﾜﾝﾟ]+)')
	ABCs = re.compile("[A-Za-z]")
	kigou= re.compile('[!！(（）)!%$♪&/^/.]')
	sentence=sentence.replace('\r\n', '')
	sentence=sentence.replace('\n', '')
	sentence=sentence.replace('\u3000','')



	word_vector_sfc=[]
	word_vector = []
	t=Tokenizer("User_Dict1", udic_enc="CP932")
	#t=Tokenizer()
	filter_char=[]
	filter_token=[CompoundNounFilter()]
	a=Analyzer(filter_char,t,filter_token)
	#a2=Analyzer(filter_char,t,[])
	for token in a.analyze(sentence):

		isMojibake = excel_moji.search(token.surface)
		isNum = num.search(token.surface)
		isKana = kana.search(token.surface)
		isABCs = ABCs.search(token.surface)
		isKigou = kigou.search(token.surface)
		if isNum != None or isABCs != None or isKana!=None or isKigou!=None or isMojibake!=None:
			continue
		elif token.base_form=='':
			continue
		#elif token.surface in stopwords:
			#continue

		elif token.part_of_speech.split(',')[1] in ['複合']:
			t=0
			complex_surface_list=[]
			complex_surface=''
			for token2 in a.analyze(token.surface):
				if '数' in token2.part_of_speech.split(',')[1]:
					continue

				else:
					complex_surface+=token2.base_form
					complex_surface_list.append(token2.base_form)
					t+=1

			if t<3:
				word_vector += [complex_surface]
				word_vector_sfc += [complex_surface]
			else:
				word_vector += complex_surface_list
				word_vector_sfc += complex_surface_list
		else:
			word_vector += [token.base_form]
			word_vector_sfc += [token.surface]
	return word_vector,word_vector_sfc

def nofilter(sentence):


	excel_moji = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
	word_vector_sfc=[]
	word_vector = []
	t=Tokenizer("User_Dict1", udic_enc="CP932")
	#t=Tokenizer()
	filter_char=[]
	filter_token=[CompoundNounFilter()]
	a=Analyzer(filter_char,t,filter_token)
	#a2=Analyzer(filter_char,t,[])

	for token in a.analyze(sentence):
		isMojibake = excel_moji.search(token.surface)
		if isMojibake!=None:
			continue
		elif token.part_of_speech.split(',')[1] in ['複合']:
			t=0
			complex_surface_list=[]
			complex_surface=''
			for token2 in a.analyze(token.surface):
				if '数' in token2.part_of_speech.split(',')[1]:
					continue

				else:
					complex_surface+=token2.base_form
					complex_surface_list.append(token2.base_form)
					t+=1

			if t<3:
				word_vector += [complex_surface]
				word_vector_sfc += [complex_surface]
			else:
				word_vector += complex_surface_list
				word_vector_sfc += complex_surface_list
		else:
			word_vector += [token.base_form]
			word_vector_sfc += [token.surface]
	return word_vector,word_vector_sfc

def tokenizer_customDic(sentence):

	#csv_input = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
	#stopwords=csv_input['stopwords'].values.tolist()

	word_vector = []
	excel_moji = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
	num = re.compile("([0-9０-９])")
	kana = re.compile('([｡ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊｲﾌﾍﾎﾏﾐﾑﾒﾓﾗﾘﾙﾚﾛﾔﾕﾖﾜﾝﾟ]+)')
	ABCs = re.compile("[A-Za-z]")
	kigou= re.compile('[!！(（）)!%$♪&/^/.]')
	sentence=sentence.replace('\r\n', '')
	sentence=sentence.replace('\n', '')
	sentence=sentence.replace('\u3000','')
	#tokenizerメソッドの初期化（コンストラクタ）にユーザー辞書を指定することができる
	#ユーザー辞書は、.csvのままでもよいし、コンパイル済の辞書でもよい
	#csvを使う場合は、拡張子を付けること。
	#t=Tokenizer("User_Dict1",udic_enc="utf-8",mmap=True)
	t=Tokenizer("User_Dict1", udic_enc="CP932")
	#t=Tokenizer()
	filter_char=[]
	filter_token=[CompoundNounFilter(),POSKeepFilter(['名詞'])]

	a=Analyzer(filter_char,t,filter_token)
	#a2=Analyzer(filter_char,t,[])
	for token in a.analyze(sentence):

		isNum = num.search(token.surface)
		isKana = kana.search(token.surface)
		isABCs = ABCs.search(token.surface)
		isKigou = kigou.search(token.surface)
		isMojibake = excel_moji.search(token.surface)

		if isNum != None or isABCs != None or isKana!=None or isKigou!=None or isMojibake!=None:
			continue
		elif token.base_form=='':
			continue
		#elif token.surface in stopwords:
		#	continue
		elif token.part_of_speech.split(',')[1] in ['複合']:
			t=0
			complex_base_form=''
			for token2 in a.analyze(token.surface):
				if '数' in token2.part_of_speech.split(',')[1]:
					continue
				else:
					complex_base_form+=token2.base_form
				t+=1

			if t<3:
				word_vector += [complex_base_form]
		elif token.part_of_speech.split(',')[1] == '数':
			continue

		else:
			word_vector += [token.base_form]
		#print(token.base_form)
	return word_vector


def nofilter_eng(sentence):
	part_of_speech_rev = []
	words = nltk.word_tokenize(sentence)
	tags = nltk.pos_tag(words,tagset='universal')
	stop_words = set(stopwords.words('english'))
	#select_stop_words = set(('castle','nijo', 'went','NijoCastle','There','go',"n't",'Gion','Arashiyama',"'s","i","by",'\\','from','also','In','when','for','so','be','not'))
	#print(tags)
	for rev in tags:
		if rev[1] == 'NOUN':
			word_l = rev[0].lower()
			print(rev)
			if not word_l in stop_words or select_stop_words:
				#selected_word_rev=selected_word_rev.replace('.','').replace('!','').replace(':','').replace(',','').replace('__','').replace('・','').replace('(','').replace(')','')
				#selected_word_rev=selected_word_rev.remove('')
				part_of_speech_rev.append(rev[0])
				#print(len(stop_words))
		#elif rev[1] == 'ADJ':
		#	print(rev)
		#	if not rev[0] in stop_words or select_stop_words:
				#selected_word_rev=selected_word_rev.replace('.','').replace('!','').replace(':','').replace(',','').replace('__','').replace('・','').replace('(','').replace(')','')
				#selected_word_rev=selected_word_rev.remove('')
		#		part_of_speech_rev.append(rev[0])
		else:
			#print(stop_words)
			continue
    #print([word for word, tag in select_tag])
	print(part_of_speech_rev)
	return part_of_speech_rev



def filter_eng(sentence):
	part_of_speech_rev = []
	words = nltk.word_tokenize(sentence)
	tags = nltk.pos_tag(words)
	stop_words = set(stopwords.words('english'))
	csv_input = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
	select_stop_words=set(csv_input['stopwords'].values.tolist())
	#select_stop_words = set(("kyoto","tokyo","u","'",'"',".","arashiyama","ginza","asakusa","sensoji","ueno","ameyoko","roppongi","odaiba","shibuya","gion"))
	for rev in tags:
		if rev[1] == 'NNP' or rev[1] == 'NN':
			#print('true',rev[1])
			word_l = rev[0].lower()
			#print(rev,word_l)
			if not word_l in stop_words and not word_l in select_stop_words:
				if type(word_l) is bool:
					print('false')
				else:
					print("word_l",word_l)
					part_of_speech_rev.append(word_l)
		elif rev[1] == 'NNS' or rev[1] == 'NNPS':
			word_l = rev[0].lower()
			p = inflect.engine()
			word_s = p.singular_noun(word_l)
			if not word_s in stop_words and not word_s in select_stop_words:
				if type(word_s) is bool:
					print('false')
				else:
					print("word_s",word_s)
					part_of_speech_rev.append(word_s)
		#elif rev[1] == 'JJS' or rev[1] == 'JJR' or rev[1] == 'JJ':
		#	word_l = rev[0].lower()
		#	if not word_l in stop_words and not word_l in select_stop_words:
		#		if type(word_l) is bool:
		#			print('false')
		#		else:
		#			print("word_noun",word_l)
		#			part_of_speech_rev.append(word_l)
		else:
			#print(stop_words)
			continue
    #print([word for word, tag in select_tag])
	print(part_of_speech_rev)
	return part_of_speech_rev

# convert to lower case
#tokens = [w.lower() for w in tokens]
# remove punctuation from each word
#import string
#table = str.maketrans('', '', string.punctuation)
#stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
#words = [word for word in stripped if word.isalpha()]
# filter out stop words
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#words = [w for w in words if not w in stop_words]
#print(words[:100])


def filter_eng_inc_verb(sentence):
	part_of_speech_rev = []
	words = nltk.word_tokenize(sentence)
	tags = nltk.pos_tag(words)
	stop_words = set(stopwords.words('english'))
	csv_input = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
	selected_stop_words=set(csv_input['stopwords'].values.tolist())
	#ps = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	for rev in tags:
		rev_word = rev[0].replace('u3000', '').replace('%', '').replace('/', '').replace('...', '')
		if rev[1] == 'NNP' or rev[1] == 'NN':
			#print('true',rev[1])
			word_n = rev_word.lower()
			#print("word_n is.......",word_n)
			word_n_sfc = lemmatizer.lemmatize(word_n)
			#print("word_n_sfc is.......",word_n_sfc)
			#print(rev,word_l)
			if not word_n in stop_words and not word_n in selected_stop_words:
				if type(word_n) is bool:
					print('pass bool')
				else:
					part_of_speech_rev.append(word_n)
					#print("word_l",word_l)

		elif rev[1] == 'NNS' or rev[1] == 'NNPS':
			word_n_p = rev_word.lower()
			#print("word_n_p is.......",word_n_p)
			#p = inflect.engine()
			#word_n_sfc = p.singular_noun(word_n_p)
			word_n_p_sfc = lemmatizer.lemmatize(word_n_p)
			#print("word_n_p_sfc is.......",word_n_p_sfc)
			if not word_n_p_sfc in stop_words and not word_n_p_sfc in selected_stop_words:
				if type(word_n_p_sfc) is bool:
					print('pass bool')
				else:
					#print("word_s",word_s)
					part_of_speech_rev.append(word_n_p_sfc)

		elif rev[1] == 'JJS' or rev[1] == 'JJR' or rev[1] == 'JJ':
			word_j = rev_word.lower()
			#word_j_sfc = lemmatizer.lemmatize(word_j)
			if not word_j in stop_words and not word_j in selected_stop_words:
				if type(word_j) is bool:
					print('pass bool')
				else:
					#print("word_adjective",word_j)
					part_of_speech_rev.append(word_j)

		elif rev[1] == 'VB' or rev[1] == 'VBD' or rev[1] == 'VBG'or rev[1] == 'VBN' or rev[1] == 'VBP' or rev[1] == 'VBZ':
			word_v = rev_word.lower()
			#print("word_v is.......",word_v)
			word_v_sfc = lemmatizer.lemmatize(word_v, pos = 'v')
			# get rid of it after downloading wordnet & wordnet_ic
			#print(type(word_v_sfc))
			#print("word_v_sfc is.......",word_v_sfc)
			if not word_v_sfc in stop_words and not word_v_sfc in selected_stop_words:
				if type(word_v) is bool:
					print('pass bool')
				else:
					part_of_speech_rev.append(word_v_sfc)
		else:
			#print("not in these tags")
			continue

	print(part_of_speech_rev)
	return part_of_speech_rev


def filter_eng_new_non_stopwords(sentence):
	part_of_speech_rev = []
	words = nltk.word_tokenize(sentence)
	tags = nltk.pos_tag(words)
	stop_words = set(stopwords.words('english'))
	ps = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	for rev in tags:
		rev_word = rev[0].replace('u3000', '')
		if rev[1] == 'NNP' or rev[1] == 'NN':
			#print('true',rev[1])
			word_n = rev_word.lower()
			print("word_n is.......",word_n)
			word_n_sfc = lemmatizer.lemmatize(word_n)
			print("word_n_sfc is.......",word_n_sfc)
			#print(rev,word_l)
			if not word_n in stop_words:
				if type(word_n) is bool:
					print('pass bool')
				else:
					part_of_speech_rev.append(word_n)
					#print("word_l",word_l)

		elif rev[1] == 'NNS' or rev[1] == 'NNPS':
			word_n_p = rev_word.lower()
			print("word_n_p is.......",word_n_p)
			#p = inflect.engine()
			#word_n_sfc = p.singular_noun(word_n_p)
			word_n_p_sfc = lemmatizer.lemmatize(word_n_p)
			print("word_n_p_sfc is.......",word_n_p_sfc)
			if not word_n_p_sfc in stop_words:
				if type(word_n_p_sfc) is bool:
					print('pass bool')
				else:
					#print("word_s",word_s)
					part_of_speech_rev.append(word_n_p_sfc)

		elif rev[1] == 'JJS' or rev[1] == 'JJR' or rev[1] == 'JJ':
			word_j = rev_word.lower()
			#word_j_sfc = lemmatizer.lemmatize(word_j)
			if not word_j in stop_words:
				if type(word_j) is bool:
					print('pass bool')
				else:
					#print("word_adjective",word_j)
					part_of_speech_rev.append(word_j)

		elif rev[1] == 'VB' or rev[1] == 'VBD' or rev[1] == 'VBG'or rev[1] == 'VBN' or rev[1] == 'VBP' or rev[1] == 'VBZ':
			word_v = rev_word.lower()
			print("word_v is.......",word_v)
			word_v_sfc = lemmatizer.lemmatize(word_v, pos = 'v')
			#you will get rid of it after downloading wordnet & wordnet_ic
			print(type(word_v_sfc))
			print("word_v_sfc is.......",word_v_sfc)
			if not word_v_sfc in stop_words:
				if type(word_v) is bool:
					print('pass bool')
				else:
					part_of_speech_rev.append(word_v_sfc)
		else:
			print("not in these tags")
			continue

	#print(' '.join(part_of_speech_rev))
	return ' '.join(part_of_speech_rev), part_of_speech_rev

def filter_eng_simple(sentence):
	part_of_speech_rev = []
	words = nltk.word_tokenize(sentence)
	part_of_speech_rev.append(words)
	#print(part_of_speech_rev)
	return part_of_speech_rev


def filter_eng_only_noun(sentence):
	part_of_speech_rev = []
	sentence_replace = sentence.replace('.', ' ').replace('/', ' ').replace('...', ' ')
	words = nltk.word_tokenize(sentence_replace)
	tags = nltk.pos_tag(words,tagset='universal')
	stop_words = stopwords.words('english')
	csv_input = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
	selected_stop_words = csv_input['stopwords'].values.tolist()
	#excel_moji = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
	num = re.compile(r'([0-9０-９]+)')
	kana = re.compile(r'([｡ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊｲﾌﾍﾎﾏﾐﾑﾒﾓﾗﾘﾙﾚﾛﾔﾕﾖﾜﾝﾟ]+)')
	kanji = re.compile(r'([ぁ-んァ-ヶ亜-熙]+)')
	kigou= re.compile(r'([!！(（）)!%$♪&/^/.<>＜＞]+)')
	# sentence=sentence.replace('\r\n', '')
	# sentence=sentence.replace('\n', '')
	# sentence=sentence.replace('\u3000','')
	#ps = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	for rev in tags:
		rev_word = rev[0]
		#print(rev_word)
		
		rev_word = rev_word.lower()
		rev_word = lemmatizer.lemmatize(rev_word)

		find_stop_words = rev_word in stop_words
		find_selected_stop_words = rev_word in selected_stop_words

		isNum = re.findall(num, rev_word)
		isKana = re.findall(kana, rev_word)
		isKigou = re.findall(kigou, rev_word)
		#isMojibake = re.findall(kigou, rev_word)
		iskanji = re.findall(kanji, rev_word)


		if rev[1] != 'NOUN':
			#print('NOT NOUN',rev_word)
			continue
		elif len(isNum) != 0 or len(isKigou) != 0 or len(isKana) != 0 or len(iskanji) != 0:
			print(rev_word)
			continue
		elif find_stop_words == True or find_selected_stop_words == True:
			#print(rev_word)
			continue
		else:
			if type(rev_word) is bool:
				print('pass bool')
			else:
				#print(rev_word)
				part_of_speech_rev.append(rev_word)
				#print("word_l",word_l)


	print(part_of_speech_rev)
	return part_of_speech_rev




if __name__ == '__main__':

	sentence = input('文を入力してください：')
	word_vector,word_vector_surface=nofilter(sentence)
	#wordvec = tokenizer_customDic(sentence)

	print('基本形',word_vector,'表層形',word_vector_surface)
