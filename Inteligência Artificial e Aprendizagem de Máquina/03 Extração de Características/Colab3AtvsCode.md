## 3. Extração Manual de Características

```python
def extrair_caracteristicas(texto):
    return {
        'num_palavras': len(texto.split()),
        'num_caracteres': len(texto),
        'tem_exclamacao': '!' in texto,
        'num_maiusculas': sum(1 for c in texto if c.isupper())
    }

exemplo = "A Inovação é FUNDAMENTAL para o sucesso!"
print(extrair_caracteristicas(exemplo))
```


## 4. Extração com N-gramas

**N-gramas** são sequências de *n* palavras consecutivas em um texto. São úteis para capturar contexto e padrões linguísticos.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "o gato preto dorme",
    "o cachorro branco corre"
]

vetorizador = CountVectorizer(ngram_range=(1,2))  # unigramas e bigramas
X = vetorizador.fit_transform(corpus)

print(vetorizador.get_feature_names_out())
print(X.toarray())
```


## 5. Extração de Características Sintáticas com spaCy

```python
import spacy
nlp = spacy.load("pt_core_news_sm")

doc = nlp("Maria comprou um carro vermelho.")

caracteristicas = [(token.text, token.pos_, token.dep_) for token in doc]
for palavra, classe, dependencia in caracteristicas:
    print(f"{palavra:<10} | {classe:<10} | {dependencia}")
```


## 6. Extração de Entidades Nomeadas (NER)

```python
texto = "O presidente Lula visitou Brasília em janeiro."

doc = nlp(texto)
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```


## 7. Características Avançadas com Word Embeddings

Embora técnicas como Bag of Words sejam simples, não capturam o significado das palavras. Para isso, utilizamos **vetores densos semânticos**, como Word2Vec, GloVe e BERT.

Exemplo com spaCy (embeddings pré-treinados):

```python
palavra1 = nlp("rei")[0]
palavra2 = nlp("rainha")[0]
print(palavra1.similarity(palavra2))
```


## 8. Aplicação: Conjunto de Características para Análise de Sentimentos

```python
def extrair_features_completas(texto):
    doc = nlp(texto)
    return {
        'num_palavras': len(doc),
        'num_substantivos': sum(1 for token in doc if token.pos_ == "NOUN"),
        'num_adjetivos': sum(1 for token in doc if token.pos_ == "ADJ"),
        'num_entidades': len(doc.ents),
        'media_similaridade': doc[0].similarity(doc[-1])
    }

texto = "A experiência foi maravilhosa, atendimento excelente e comida deliciosa!"
print(extrair_features_completas(texto))
```
