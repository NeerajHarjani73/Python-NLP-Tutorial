# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:47:23 2020

@author: neera
"""
import nltk
import os
import nltk.corpus
nltk.download()

#Tokenization -> strings to tokens
print(os.listdir(nltk.data.find("corpora")))
from nltk.corpus import brown
brown.words()
nltk.corpus.gutenberg.fileids()

hamlet = nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')
hamlet

for word in hamlet[:500]:
    print(word, sep = ' ', end = ' ')
    
hamlet_str = """The Tragedie of Macbeth by William Shakespeare 1603 ] Actus Primus . Scoena Prima . Thunder and Lightning . Enter three Witches . 1 . When shall we three meet againe ? In Thunder , Lightning , or in Raine ? 2 . When the Hurley - burley ' s done , When the Battaile ' s lost , and wonne 3 . That will be ere the set of Sunne 1 . Where the place ? 2 . Vpon the Heath 3 . There to meet with Macbeth 1 . I come , Gray - Malkin All . Padock calls anon : faire is foule , and foule is faire , Houer through the fogge and filthie ayre . Exeunt . Scena Secunda . Alarum within . Enter King Malcome , Donalbaine , Lenox , with attendants , meeting a bleeding Captaine . King . What bloody man is that ? he can report , As seemeth by his plight , of the Reuolt The newest state Mal . This is the Serieant , Who like a good and hardie Souldier fought ' Gainst my Captiuitie : Haile braue friend ; Say to the King , the knowledge of the Broyle , As thou didst leaue it Cap . Doubtfull it stood , As two spent Swimmers , that doe cling together , And choake their Art : The mercilesse Macdonwald ( Worthie to be a Rebell , for to that The multiplying Villanies of Nature Doe swarme vpon him ) from the Westerne Isles Of Kernes and Gallowgrosses is supply ' d , And Fortune on his damned Quarry smiling , Shew ' d like a Rebells Whore : but all ' s too weake : For braue Macbeth ( well hee deserues that Name ) Disdayning Fortune , with his brandisht Steele , Which smoak ' d with bloody execution ( Like Valours Minion ) caru ' d out his passage , Till hee fac ' d the Slaue : Which neu ' r shooke hands , nor bad farwell to him , Till he vnseam ' d him from the Naue toth ' Chops , And fix ' d his Head vpon our Battlements King . O valiant Cousin , worthy Gentleman Cap . As whence the Sunne ' gins his reflection , Shipwracking Stormes , and direfull Thunders : So from that Spring , whence comfort seem ' d to come , Discomfort swells : Marke King of Scotland , marke , No sooner Iustice had , with Valour arm ' d , Compell ' d these skipping Kernes to trust their heeles , But the Norweyan Lord , surueying vantage , With furbusht Armes , and new supplyes of men , Began a fresh assault King . Dismay ' d not this our Captaines , Macbeth and Banquoh ? Cap . Yes , as Sparrowes , Eagles ; Or the Hare , the Lyon : If I say sooth , I must report"""
type(hamlet_str)

from nltk.tokenize import word_tokenize
tokens = word_tokenize(hamlet_str)
tokens
len(tokens)

from nltk.probability import FreqDist
fdist = FreqDist()
for word in tokens:
    fdist[word.lower()]+=1
fdist
len(fdist)

#Create tuples of tokens
from nltk.util import bigrams,trigrams,ngrams
hamlet_bigrams = list(nltk.bigrams(tokens))
hamlet_bigrams

hamlet_bigrams = list(nltk.trigrams(tokens))
hamlet_bigrams

hamlet_bigrams = list(nltk.ngrams(tokens,5))
hamlet_bigrams

#Stemming
from nltk.stem import PorterStemmer
pst = PorterStemmer()

pst.stem('having')

words_to_stem = ['give','giving','given','gave']
for words in words_to_stem:
    print(words+': '+pst.stem(words))

from nltk.stem import LancasterStemmer
lst = LancasterStemmer()
for words in words_to_stem:
    print(words+': '+lst.stem(words))
    
from nltk.stem import SnowballStemmer
sbst = SnowballStemmer('english')
for words in words_to_stem:
    print(words+': '+sbst.stem(words))
    
#Stemming not always work, let's use Lemmatization

from nltk.stem import wordnet #Dictionary
from nltk.stem import WordNetLemmatizer
word_lem=WordNetLemmatizer()

word_lem.lemmatize('corpora')

for words in words_to_stem:
    print(words+': '+word_lem.lemmatize(words))
    
#Stop Words -> not useful for NLP
from nltk.corpus import stopwords
stopwords.words('english')
len(stopwords.words('english'))

import re
punctuation = re.compile(r'[-.''?!,:;()|0-9]')

post_punctuation = []
for words in tokens:
    word=punctuation.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)
        
#POS: Tags and Descriptions (Noun,verb,...)
sent = "Timothy is a natural when it comes to drawing"
sent_tokens = word_tokenize(sent)
for token in sent_tokens:
    print(nltk.pos_tag([token]))
    
#Name and entity recognition 
from nltk import ne_chunk
NE_sent= "The US President stays in the White House"
NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)
NE_NER = ne_chunk(NE_tags)
print(NE_NER)    

from nltk import RegexpParser
new = "The big cat ate the little mouse who was after fresh cheese"
new_tokens = nltk.pos_tag(word_tokenize(new))
new_tokens
grammar_np = r"NP:{<Dt>?<JJ>*<NN>}"
chunk_parser = RegexpParser(grammar_np)
chunk_result = chunk_parser.parse(new_tokens)
chunk_result

