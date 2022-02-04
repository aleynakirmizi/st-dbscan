import pandas as pd
import time
import re
import nltk
from st_dbscan import st_dbSCAN
import numpy as np
import json
Series=pd.read_json('News_Category_Dataset_v2.json',lines=True)
df = pd.DataFrame(Series)
print(df.columns)
df2=df.head(1000)
train=df2[['category','headline','date']]
train['date']= train['date'].apply(lambda x: time.mktime(x.timetuple()))
train =train.head(1000)
import string
nltk.download('punkt')
nltk.download('stopwords')
def clean_text(text):
    text = text.lower()                                  # lower-case all characters
    text =  re.sub(r'@\S+', '',text)                     # remove twitter handles
    text =  re.sub(r'pic.\S+', '',text)
    text =  re.sub(r"[^a-zA-Z+']", ' ',text)             # only keeps characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')      # keep words with length>1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i)>2])
    text= re.sub("\s[\s]+", " ",text).strip()            # remove repeated/leading/trailing spaces
    return text


train['Text_cleaning'] = train.headline.apply(clean_text)
train['tokenized_sents'] = train.apply(lambda row: nltk.word_tokenize(row['Text_cleaning']), axis=1)
data=train.loc[:,['date']].values
df= pd.DataFrame()
df["vector_text"] = train['tokenized_sents']
df["date"] = data
data =(data-np.min(data))/(np.max(data)-np.min(data))
df["date"] = data*100


st = st_dbSCAN(df,0.8,50,5,5)
ft=st.fit_transform(df)
print(df.cluster.value_counts())


