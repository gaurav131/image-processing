from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import ngrams
import re
corp = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
pattern = r"[^A-Za-z]"
a = "hey how are you doing, are you practising python these days? and running can be really hard"
a = re.sub(pattern, " ", a)
# print(a)
tokenizer = WordPunctTokenizer()
seperated = tokenizer.tokenize(a)
# print(seperated)
# print([lemmatizer.lemmatize(x) for x in seperated if x not in corp])
# print(list(ngrams(seperated, 6)))
b = TfidfVectorizer()
b.fit_transform(seperated)
print(b.vocabulary_)