from sklearn.manifold import TSNE
import pandas as pd
import time
import re
import nltk
from st_dbscan import st_dbSCAN
import numpy as np
import json
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import silhouette_score


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
vectorizer = CountVectorizer()
data_vectorizer = vectorizer.fit_transform(train['Text_cleaning'])
arr=data_vectorizer.todense()
train['vectorized_text']=arr.tolist()
train['tokenized_sents'] = train.apply(lambda row: nltk.word_tokenize(row['Text_cleaning']), axis=1)
data=train.loc[:,['date']].values
df= pd.DataFrame()
df["vector_text"] = train['tokenized_sents']
# df["date"] = data
# data =(data-np.min(data))/(np.max(data)-np.min(data))
df["date"] = data
# print(df)
from sklearn.metrics import silhouette_score
# list1 = [50,55,60,65]
# minpt=[3,4,5,6]
# for i in list1:
#     for j in minpt:
#         st = st_dbSCAN(df, 0.8, i, j, 5)
#         ft = st.fit_transform(df)
#         score = silhouette_score(train['vectorized_text'].values.tolist(), df["cluster"].values)
#         print("eps2: ",i," minpt: ",j," score: ",score)
#         print("----------------")
#         print(df.cluster.value_counts())
#         print("-------------")
# epsilons=[3,4,5,6,7,8]
# for epsilon in epsilons :
#     st = st_dbSCAN(df, 0.8, 60, 8, epsilon)
#     ft = st.fit_transform(df)
#     score = silhouette_score(train['vectorized_text'].values.tolist(), df["cluster"].values)
#     print("epsilon: ",epsilon,"score: ",score)
#     print("----------------")
#     print(df.cluster.value_counts())
#     print("------------------")

list1 =[0.8,0.9]
list2= [90000,100000,120000,150000]
list3 = [6,7,8]
list4=[4,5,6,7]
for i in list1:
    for j in list2:
        for k in list3:
            for m in list4:
                st = st_dbSCAN(df, i, j,k,m)
                ft = st.fit_transform(df)
                score = silhouette_score(train['vectorized_text'].values.tolist(), df["cluster"].values)
                print(" eps1: ",i," eps2: ",j," minpoint: ",k," epsilon:",m," score:",score)
# st = st_dbSCAN(df, 0.8,90000, 7, 6)
# ft = st.fit_transform(df)
# score = silhouette_score(train['vectorized_text'].values.tolist(), df["cluster"].values)
# print("score: ",score)


