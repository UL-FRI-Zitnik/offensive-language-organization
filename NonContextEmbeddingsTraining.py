#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import modules & set up logging
from gensim.models import Word2Vec, FastText
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
import os
import re
import stanza
stanza.download('en', processors='tokenize')
nlp = stanza.Pipeline('en', processors='tokenize')


# In[ ]:


class OffensiveSentences(object):
    def __init__(self):
        pass
 
    def __iter__(self):
        def special_chars_or_punctuation_only(word):
            word = re.sub('[^a-zA-Z:]', '', word)          # Remove punctuations
            word = re.sub("(\\d|\\W)+","",word)            # remove special characters and digits
            return len(word) == 0

        for line in open(os.path.join('./full_textOnly_cleaned_dataset.csv'), encoding="utf-8"):
            line = re.sub('\n', '', line)
            #print(f"ORIGINAL: '{line}'")            
            line = line.lower()                             # Convert to lowercase
            line = re.sub(r'\s+',' ', line)                  # Remove duplicated whitespaces
            processed_line = [word.text for sentence in nlp(line).sentences for word in sentence.words if not special_chars_or_punctuation_only(word.text)]            
            #print(f"PROCESSED: '{processed_line}'")
            yield processed_line
 
sentences = OffensiveSentences() # a memory-friendly iterator


# In[ ]:


model = Word2Vec(sentences, min_count=2, vector_size=50, workers=16)


# In[ ]:


model.save('w2v_50dim_model_v2')
new_model = gensim.models.Word2Vec.load('w2v_50dim_model_v2')


# In[ ]:


model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)


# In[ ]:


model.doesnt_match("breakfast cereal dinner lunch".split())


# In[ ]:


model.similarity('woman', 'man')


# In[ ]:


model['computer']


# In[ ]:


model.wv.most_similar('computer', topn=10)


# In[ ]:


model = FastText(sentences, min_count=2, vector_size=50, workers=4)
model.save('fastText_50dim_model_v2')


# In[ ]:





# In[ ]:





# In[ ]:




