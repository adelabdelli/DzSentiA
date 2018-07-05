#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import re
from os import listdir
from os.path import isfile, join
from pyarabic import araby

def clean(text):
    text = text.replace("<br/>", " ")
    strip_special_chars = re.compile(u'[^\u0621-\u064a ]')
    return re.sub(strip_special_chars, " ", text)

def process(text):
	text = araby.strip_tashkeel(text) #delete *tashkil
	
	text = re.sub('\ـ+', ' ', text)  # delete letter madda
	
	text = re.sub('\ر+', 'ر', text)  # duplicate ra2
	text = re.sub('\اا+','ا',text)     #duplicate alif
	text = re.sub('\ووو+','و',text)    #duplicate waw (more than 3 times goes to 1
	text = re.sub('\ههه+','ههه',text)  #duplicate ha2 (more than 3 times goes to 1
	text = re.sub('\ةة+','ة',text)
	text = re.sub('\ييي+','ي',text)
	text = re.sub('أ','ا',text) # after to avoid mixing
	text = re.sub('آ','ا',text) # after to avoid mixing
	text = re.sub('إ','ا',text) # after to avoid mixing
	text = re.sub('ة','ه',text) # after ةة to avoid mixing ههه
	text = re.sub('ى','ي',text)
	
	text = " ".join(text.split()) #delete multispace
	
	return text


