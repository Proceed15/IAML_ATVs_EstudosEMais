### Pré-processamento com NLTK

```python
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocessar_texto(texto):
    tokens = word_tokenize(texto.lower())
    stop_words = set(stopwords.words('portuguese'))
    tokens_filtrados = [
        palavra for palavra in tokens 
        if palavra not in stop_words and palavra not in string.punctuation
    ]
    return tokens_filtrados

texto = "Este é um exemplo simples de pré-processamento de texto em PLN."
print(preprocessar_texto(texto))
```


### Vetorização com Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "O gato preto dorme no sofá",
    "O cachorro branco corre no parque"
]

vetorizador = CountVectorizer()
X = vetorizador.fit_transform(corpus)

print(vetorizador.get_feature_names_out())
print(X.toarray())
```


### Vetorização com TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vetorizador_tfidf = TfidfVectorizer()

X_tfidf = vetorizador_tfidf.fit_transform(corpus)

print(vetorizador_tfidf.get_feature_names_out())

print(X_tfidf.toarray())
# The Secondary