## 3. Exemplo Prático – Abordagem Léxica

```python
lexico = {
    "ótimo": 1,
    "bom": 1,
    "excelente": 1,
    "ruim": -1,
    "horrível": -1,
    "péssimo": -1
}

def analisar_sentimento_simples(texto):
    tokens = texto.lower().split()
    score = sum(lexico.get(palavra, 0) for palavra in tokens)
    if score > 0:
        return "positivo"
    elif score < 0:
        return "negativo"
    return "neutro"

exemplo = "O produto é excelente, mas o atendimento foi ruim"
print(analisar_sentimento_simples(exemplo))
```


## 4. Abordagem com Machine Learning

### 4.1 Base de Dados de Exemplo

```python
corpus = [
    "O filme foi excelente e muito divertido",  # positivo
    "O atendimento foi péssimo",                # negativo
    "Gostei do serviço prestado",               # positivo
    "Não recomendo este lugar",                 # negativo
    "Foi bom, mas poderia ser melhor",          # neutro
]

rotulos = ["positivo", "negativo", "positivo", "negativo", "neutro"]
```

### 4.2 Treinamento com Naive Bayes

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(corpus, rotulos, test_size=0.4)

modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 5. Avaliação de Modelos

As métricas mais comuns utilizadas na avaliação de classificadores de sentimento incluem:

* **Acurácia**
* **Precisão**
* **Revocação**
* **F1-score**

A escolha da métrica depende do equilíbrio entre classes e dos objetivos do sistema.

## 6. Desafios Comuns

* Ambiguidade: “Este filme é tão ruim que chega a ser bom.”
* Ironia e sarcasmo: “Adorei esperar 3 horas na fila...”
* Gírias e expressões regionais
* Palavras fora do vocabulário do modelo (Out-of-Vocabulary)

## 7. Aplicação com Biblioteca TextBlob (opcional)

```python
from textblob import TextBlob
frase = TextBlob("O show foi incrível e emocionante.")
print(frase.sentiment)  # polarity e subjectivity
```

> Nota: `TextBlob` funciona melhor com textos em inglês. Para português, há bibliotecas como `VADER-PT` ou uso de modelos pré-treinados em português.
