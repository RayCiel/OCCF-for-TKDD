import nltk
import string
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import MySQLdb
import time
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation



db = MySQLdb.connect("localhost", "root", "123456", "twitter1", charset='utf8' )
cursor = db.cursor()

def textPrecessing(text):
    text = text.lower()
    for c in string.punctuation:
        text = text.replace(c, ' ')
    wordLst = nltk.word_tokenize(text)
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]

    return " ".join(filtered)
with open("user_id.csv", 'r') as in_csv:
    tmp = csv.reader(in_csv)
    column1 = [row for row in tmp]
user_id = [i[0] for i in column1]
with open("status_id.csv", 'r') as in_csv:
    tmp = csv.reader(in_csv)
    column1 = [row for row in tmp]
status_id_r = [i[0] for i in column1]
print(len(status_id_r))
sql = "SELECT * FROM twitter1.status"
status_text = []
status_text_tmp = []
status_id_tmp = []
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
    if (row[0] in status_id_r and row[1] in user_id):
        status_text_tmp.append(row[2])
        status_id_tmp.append(row[0])
for i in range(len(status_id_r)):
    index = status_id_tmp.index(status_id_r[i])
    status_text.append(status_text_tmp[index])
docLst = []
print(len(status_text))
print(len(status_id_r))
for desc in status_text :
    docLst.append(textPrecessing(desc).encode('utf-8'))
with open("data", 'w') as f:
    for line in docLst:
        f.write(str(line)+'\n')

#docLst = []
#with open("data", 'r') as f:
#    for line in f.readlines():
#        if line != '':
#            docLst.append(line.strip())
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=10000,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(docLst)
joblib.dump(tf_vectorizer,"ldamodel" )
n_topics = 1000
lda = LatentDirichletAllocation(n_topics=n_topics   ,
                                max_iter=50,
                                learning_method='batch')
lda.fit(tf)
doc_topic_dist = lda.transform(tf)
np.save("lda.npy",doc_topic_dist)
print("done!")
