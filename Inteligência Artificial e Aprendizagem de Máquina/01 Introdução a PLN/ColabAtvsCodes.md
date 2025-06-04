### Tokenização e Remoção de Stopwords

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

texto = "O Processamento de Linguagem Natural permite que computadores entendam a linguagem humana."
tokens = word_tokenize(texto, language='portuguese')
stop_words = set(stopwords.words('portuguese'))

tokens_filtrados = [palavra for palavra in tokens if palavra.lower() not in stop_words]

print("Tokens:", tokens)
print("Tokens sem stopwords:", tokens_filtrados)

### Stemming
```python
from nltk.stem import RSLPStemmer
nltk.download('rslp')
stemmer = RSLPStemmer()

palavras = ['correndo', 'correu', 'corre', 'corrida']
for p in palavras:
    print(f"{p} → {stemmer.stem(p)}")
```



### Etiquetagem Gramatical (POS Tagging)
```python

nltk.download('mac_morpho')
from nltk.corpus import mac_morpho
from nltk import UnigramTagger

train_sents = mac_morpho.tagged_sents()[:1000]
tagger = UnigramTagger(train_sents)
print(tagger.tag(['Maria', 'comprou', 'um', 'carro', 'novo']))
```

### Reconhecimento de Entidades Nomeadas (NER) com spaCy

```python
!pip install -U spacy
!python -m spacy download pt_core_news_sm

import spacy
nlp = spacy.load("pt_core_news_sm")

doc = nlp("A IBM foi fundada em 1911 e está sediada em Nova Iorque.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

# All the Codes of the First Class

