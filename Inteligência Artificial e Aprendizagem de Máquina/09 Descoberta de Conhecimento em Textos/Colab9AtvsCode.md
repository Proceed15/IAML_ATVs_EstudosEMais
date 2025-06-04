Vamos usar a biblioteca **spaCy**, que é eficiente e muito usada para PLN e tarefas de IE.

#### 1. Instalar spaCy e carregar modelo

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### 2. Código para NER (Reconhecimento de Entidades Nomeadas)

```python
import spacy

# Carregar o modelo de idioma em inglês
nlp = spacy.load("en_core_web_sm")

# Texto de exemplo
text = "Elon Musk is the CEO of SpaceX and was born in Pretoria, South Africa."

# Processar o texto
doc = nlp(text)

# Exibir entidades nomeadas
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
```

#### Saída Esperada:

```
Elon Musk (PERSON)
SpaceX (ORG)
Pretoria (GPE)
South Africa (GPE)
```

#### 3. Visualização com displacy

```python
from spacy import displacy

displacy.serve(doc, style="ent")
```

Isso abrirá um servidor local com uma visualização colorida das entidades no navegador.

---

### Extração de Relações (Simples com regras)

```python
for token in doc:
    if token.dep_ == "ROOT":
        subject = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        object_ = [w for w in token.rights if w.dep_ in ("dobj", "attr")]
        if subject and object_:
            print(f"'{subject[0]}' {token.lemma_} '{object_[0]}'")
```

#### Exemplo com texto:

```python
text = "Jeff Bezos founded Amazon in 1994."
```

#### Saída simplificada:

```
'Jeff Bezos' found 'Amazon'
```

---

### Alternativas e Complementos

* `nltk`: bom para tarefas básicas e estudo.
* `stanza` (Stanford NLP): mais preciso, especialmente em outras línguas.
* `transformers` com modelos como `bert-base-cased` + `bertforTokenClassification` (HuggingFace) para NER e RE de alto desempenho.
* `spaCy + Matcher/EntityRuler`: regras personalizadas para detectar entidades específicas do domínio.

---

### Aplicações em Diferentes Áreas

| Área      | Aplicação de IE                                   |
| --------- | ------------------------------------------------- |
| Saúde     | Extrair sintomas, diagnósticos, tratamentos       |
| Direito   | Identificar partes, datas e termos em sentenças   |
| Negócios  | Detectar menções de concorrentes ou produtos      |
| Educação  | Mapear autores e conceitos em artigos científicos |
| Segurança | Monitoramento de ameaças em redes sociais         |

---

Vamos usar o dataset `20newsgroups`, que contém textos de fóruns categorizados em 20 tópicos.

#### 1. Instalar bibliotecas necessárias

```bash
pip install scikit-learn
```

#### 2. Código completo

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Carregar dados
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.hockey', 'talk.politics.mideast'])

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Criar pipeline com TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treinar o modelo
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

#### Exemplo de predição com novo texto:

```python
text = ["NASA discovered a new planet close to the solar system."]
print(model.predict(text))  # Saída: índice da categoria
print(data.target_names[model.predict(text)[0]])  # Nome da categoria
```

---

### Alternativas de Modelos

* **Tradicionais**:

  * Naive Bayes
  * SVM
  * Logistic Regression
* **Deep Learning**:

  * Redes neurais com LSTM, CNN
  * Transformers (como BERT, RoBERTa, DistilBERT)

---

### Exemplo com Transformers (Hugging Face)

```bash
pip install transformers torch
```

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Exemplo
print(classifier("This movie was absolutely fantastic!"))
```

---

### Exemplo Prático em Python (K-Means com TF-IDF)

#### 1. Instale as dependências

```bash
pip install scikit-learn matplotlib
```

#### 2. Código completo

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Coleta de dados
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.hockey', 'talk.politics.mideast'])
texts = data.data

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X = vectorizer.fit_transform(texts)

# Aplicar K-Means com 3 clusters
k = 3
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# Plot dos clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=model.labels_, cmap='rainbow', alpha=0.6)
plt.title("Agrupamento de documentos com K-Means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
```

---

### Exibindo Palavras Representativas dos Clusters

```python
import numpy as np

terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

print("\nPalavras principais por cluster:")
for i in range(k):
    print(f"\nCluster {i}:")
    for j in order_centroids[i, :10]:
        print(f"- {terms[j]}")
```

---

### Variações Avançadas

* Clustering com **BERT embeddings** e **UMAP** para reduzir dimensionalidade.
* **Clustering hierárquico** para visualizar fusões entre documentos em forma de dendrograma.
* Uso de **Top2Vec** ou **BERTopic** para clustering + extração de tópicos automaticamente.

---

### Considerações Finais

* O clustering é útil quando **não se conhece previamente o número ou os tipos de categorias** nos dados.
* A **qualidade dos vetores** e a **escolha do algoritmo** têm grande impacto nos resultados.
* A interpretação dos clusters exige **análise qualitativa** das palavras ou documentos que compõem cada grupo.

---


#### 1. Instalação

```bash
pip install scikit-learn numpy nltk gensim pyLDAvis
```

#### 2. Código com LDA usando Gensim

```python
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Conjunto de documentos simples
documents = [
    "Artificial intelligence is transforming industry.",
    "Machine learning improves business analytics.",
    "Deep learning and neural networks are hot topics.",
    "Football is a popular sport in Europe.",
    "The World Cup is watched by millions of people.",
    "Messi and Ronaldo are legendary football players."
]

# Pré-processamento
texts = [[word.lower() for word in doc.split() if word.lower() not in stop_words] for doc in documents]

# Criação do dicionário e corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Modelagem LDA
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Exibir tópicos
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"\nTópico {idx}: {topic}")
```

---

### Visualização Interativa

```python
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)
```

---

### Saída Esperada (exemplo)

```
Tópico 0: 0.076*"football" + 0.056*"cup" + 0.055*"world" + 0.045*"popular" + 0.042*"players"
Tópico 1: 0.068*"learning" + 0.062*"deep" + 0.059*"neural" + 0.055*"networks" + 0.049*"machine"
```

---

### Alternativas Avançadas

* **BERTopic**: combina `transformers` (BERT), UMAP e clustering para gerar tópicos com **semântica profunda e interpretável**.
* **Top2Vec**: não exige definição prévia do número de tópicos.
* **Guided LDA**: permite supervisionar parcialmente os temas.

---

### Considerações

* LDA assume que cada documento é uma mistura de tópicos, e cada tópico uma distribuição de palavras.
* A escolha do número de tópicos (`num_topics`) afeta bastante o resultado — requer análise qualitativa.
* Pré-processamento de qualidade melhora muito a coerência dos tópicos.



---
### Exemplo Prático 1 – Classificação Simples com `TextBlob`

```bash
pip install textblob
python -m textblob.download_corpora
```

```python
from textblob import TextBlob

frases = [
    "I love this product! It’s amazing.",
    "The experience was terrible and disappointing.",
    "It was okay, nothing special."
]

for frase in frases:
    blob = TextBlob(frase)
    print(f"\nFrase: {frase}")
    print(f"Polaridade: {blob.sentiment.polarity:.2f}")
```

#### Interpretação:

* **polaridade > 0** → sentimento positivo
* **polaridade < 0** → sentimento negativo
* **polaridade ≈ 0** → sentimento neutro

---

### Exemplo Prático 2 – Classificação com Transformers (`transformers` + `pipeline`)

```bash
pip install transformers
```

```python
from transformers import pipeline

analisador = pipeline("sentiment-analysis")

frases = [
    "Eu adorei o atendimento e a qualidade do produto.",
    "Foi uma das piores compras que já fiz.",
    "Achei razoável, mas poderia ser melhor."
]

for frase in frases:
    resultado = analisador(frase)[0]
    print(f"\nFrase: {frase}")
    print(f"Sentimento: {resultado['label']}, Confiança: {resultado['score']:.2f}")
```

> Por padrão, usa um modelo como `distilbert-base-uncased-finetuned-sst-2-english`. Para português, podemos usar o **BERTimbau** ou **nlptown/bert-base-multilingual-uncased-sentiment**.

---

### Exemplo com Modelo em Português (via Hugging Face)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

modelo = "nlptown/bert-base-multilingual-uncased-sentiment"

analisador = pipeline("sentiment-analysis", model=modelo, tokenizer=modelo)

frase = "O produto chegou rápido e funciona perfeitamente!"
print(analisador(frase))
```

Esse modelo retorna uma **nota de 1 a 5 estrelas**, permitindo avaliações mais graduais.

---

### Desafios da Análise de Sentimentos

* **Sarcasmo e ironia**: “Que ótimo! Chegou com duas semanas de atraso.”
* **Contexto cultural e linguístico**.
* **Ambiguidade emocional**: textos com sentimentos mistos.
* **Polaridade implícita**: opiniões expressas de forma indireta.
* **Domínio específico**: modelos treinados para um domínio podem falhar em outro.

---

### Complementos Avançados

* **Aspect-Based Sentiment Analysis (ABSA)**: identifica **opiniões específicas sobre atributos**.
  Ex: “A bateria dura pouco, mas a câmera é excelente.”
* **Análise temporal de sentimentos**: como a opinião evolui ao longo do tempo.
* **Mapeamento geográfico de opiniões** (com metadados de localização).

---


### Exemplo Prático 1 – Resumo Extrativo com `sumy`

```bash
pip install sumy
```

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

texto = """
O avanço da inteligência artificial tem causado transformações profundas em diversos setores da sociedade. 
Na saúde, algoritmos ajudam no diagnóstico precoce de doenças. Na educação, sistemas adaptativos personalizam o ensino. 
Ao mesmo tempo, surgem debates éticos sobre o uso e os limites da tecnologia.
"""

parser = PlaintextParser.from_string(texto, Tokenizer("portuguese"))
summarizer = LexRankSummarizer()
resumo = summarizer(parser.document, sentences_count=2)

for frase in resumo:
    print(frase)
```

> Usa o algoritmo **LexRank**, baseado em grafos de similaridade entre sentenças.

---

### Exemplo Prático 2 – Resumo Abstrativo com Transformers (`t5-small`)

```bash
pip install transformers
```

```python
from transformers import pipeline

resumidor = pipeline("summarization", model="t5-small", tokenizer="t5-small")

texto = """
Artificial Intelligence is revolutionizing industries. In healthcare, it improves diagnostics. 
In education, it enables personalized learning. However, it also raises concerns regarding data privacy and ethical use.
"""

resumo = resumidor("summarize: " + texto, max_length=40, min_length=10, do_sample=False)

print(resumo[0]['summary_text'])
```

---

### Exemplo com Modelo em Português (Hugging Face)

Modelo recomendado: `"csebuetnlp/mT5_multilingual_XLSum"`

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

modelo = "csebuetnlp/mT5_multilingual_XLSum"

tokenizer = AutoTokenizer.from_pretrained(modelo)
modelo = AutoModelForSeq2SeqLM.from_pretrained(modelo)

resumidor = pipeline("summarization", model=modelo, tokenizer=tokenizer)

texto = """
A energia solar tem se tornado uma alternativa viável e sustentável em muitos países. 
Com a queda dos custos de instalação e o aumento da eficiência dos painéis, 
cada vez mais residências e empresas têm adotado esse tipo de energia limpa.
"""

resumo = resumidor(texto, max_length=50, min_length=15, do_sample=False)
print(resumo[0]['summary_text'])
```

---

### Desafios do Resumo Automático

* **Coerência textual**: evitar contradições ou repetições.
* **Captação da ideia central** em textos longos e não estruturados.
* **Resumo fiel** (sem distorções, especialmente em resumos abstrativos).
* **Domínio do texto**: modelos treinados em um domínio podem não funcionar bem em outro.

---


### Complementos Avançados

* **Resumo multimodal**: combinar texto + imagem + áudio.
* **Resumo guiado por pergunta** (*query-based summarization*): resumo baseado em interesse específico.
* **Resumo incremental**: para textos que crescem ao longo do tempo (ex: notícias ao vivo, chats).
---

