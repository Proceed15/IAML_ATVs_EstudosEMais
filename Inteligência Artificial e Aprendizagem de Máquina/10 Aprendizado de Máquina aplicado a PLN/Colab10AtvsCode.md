4 +  Exemplo Prático – Classificação de Sentimentos com TF-IDF + SVM

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dados de exemplo
data = {
    "texto": [
        "Adorei o filme, muito bom!",
        "Péssimo atendimento, não recomendo.",
        "Excelente experiência, voltarei!",
        "O produto chegou com defeito.",
        "Muito satisfeito com a compra."
    ],
    "sentimento": ["positivo", "negativo", "positivo", "negativo", "positivo"]
}

df = pd.DataFrame(data)

# Pré-processamento e vetorização
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['texto'])
y = df['sentimento']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LinearSVC()
model.fit(X_train, y_train)

# Previsões e avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

5 +  Uso de Embeddings Pré-treinados com Transformers

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Exemplo simples para carregar tokenizer e modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenização de exemplo
texts = ["I love this movie!", "This is terrible."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
logits = outputs.logits


**Exemplo com TF-IDF em Python:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["gosto de maçã", "não gosto de banana"]
vetorizar = TfidfVectorizer()
X = vetorizar.fit_transform(corpus)
print(X.toarray())
print(vetorizar.get_feature_names_out())
```

> *Resultado:* vetores que representam cada documento por seus termos importantes.


### 4. **Treinamento do Modelo**

Nesta fase, o modelo é treinado com os dados vetorizados.

#### Etapas:

* Divisão dos dados em **treino e teste** (geralmente 80/20 ou 70/30).
* Seleção do algoritmo: Naive Bayes, SVM, Random Forest, etc.
* Ajuste de **hiperparâmetros** com técnicas como *grid search*.

**Exemplo:**

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
modelo = MultinomialNB()
modelo.fit(X_train, y_train)
```

> *Bibliotecas:* `scikit-learn`, `XGBoost`, `PyTorch`, `TensorFlow`


### 5. **Avaliação**

Mede o desempenho do modelo com métricas apropriadas:

* **Acurácia:** proporção de acertos.
* **Precisão:** acertos sobre os classificados como positivos.
* **Recall:** cobertura dos positivos reais.
* **F1-score:** média harmônica entre precisão e recall.

**Exemplo em Python:**

```python
from sklearn.metrics import classification_report

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
```

> *Interpretação:* Métricas balanceadas são importantes, especialmente em classes desbalanceadas.

### 6. **Implantação e Ajustes**

Depois de validado, o modelo é implantado em um ambiente de produção e monitorado.
#### Etapas:
* Exportação com `joblib`, `pickle` ou `ONNX`.
* Integração com uma API (`Flask`, `FastAPI`, `Django`).
* Reavaliação periódica para ajustar **drift** (mudança nos dados).
* Implementação de **monitoramento e feedback** para aprendizado contínuo.

**Exemplo:** Um chatbot de atendimento que usa o modelo para classificar intenções e adaptar respostas automaticamente.

> *Ferramentas:* `Docker`, `CI/CD`, `MLflow`, `Streamlit`, `Gradio` para protótipos.


## 4. Exemplo Prático – Classificação de Sentimentos com TF-IDF + SVM

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = {
    "texto": [
        # Positivas
        "Adorei o atendimento, muito rápido e eficiente.",
        "Produto excelente, superou minhas expectativas!",
        "Fiquei muito satisfeito com a qualidade.",
        "O serviço foi impecável, parabéns à equipe!",
        "Compra fácil e entrega antes do prazo.",
        "Tudo certo com a encomenda, recomendo a loja.",
        "A embalagem estava perfeita e bem protegida.",
        "O suporte ao cliente foi muito prestativo.",
        "Gostei muito do produto, funciona direitinho.",
        "Muito bom, voltarei a comprar com certeza.",
        "Excelente custo-benefício.",
        "A experiência de compra foi ótima!",
        "Entrega rápida e produto de qualidade.",
        "Estou muito feliz com minha compra.",
        "Loja confiável e produtos originais.",

        # Negativas
        "Péssimo atendimento, não volto mais.",
        "Produto chegou quebrado, decepção total.",
        "Demoraram muito para entregar.",
        "A qualidade é horrível, não recomendo.",
        "Fui mal atendido e sem solução para o problema.",
        "Não funcionou como esperado, dinheiro perdido.",
        "Tive dor de cabeça com essa compra.",
        "Veio errado e ainda tive que pagar a devolução.",
        "Propaganda enganosa, produto diferente do anunciado.",
        "Sistema de entrega desorganizado.",
        "A embalagem estava rasgada e suja.",
        "O suporte não resolveu nada.",
        "Muito ruim, nunca mais compro aqui.",
        "Não cumpriram o prazo de entrega.",
        "Experiência frustrante do começo ao fim.",
    ],
    "sentimento": ["positivo"] * 15 + ["negativo"] * 15
}

df = pd.DataFrame(data)

# Vetorização
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['texto'])
y = df['sentimento']

# Divisão com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modelo
model = LinearSVC()
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```


## 5. Uso de Embeddings Pré-treinados com Transformers

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Dados simples
texts = ["I love this movie!", "This is terrible.", "Great acting!", "Worst film ever."]
labels = [1, 0, 1, 0]  # 1 = positivo, 0 = negativo

# Dataset personalizado
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenização
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset(texts, labels, tokenizer)

# Modelo
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Treinar
trainer.train()

# 4bafdaa1ba5b77b72540121a017b0b71f153908d

test_text = ["I hated this."]
inputs = tokenizer(test_text, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
pred = torch.argmax(logits, dim=1)
print("Classe prevista:", pred.item())  # 0 ou 1
```


## 6. Desafios e Considerações

* **Qualidade e quantidade dos dados**: dados ruidosos podem prejudicar o desempenho
* **Overfitting e generalização**
* **Interpretação dos modelos**
* **Custos computacionais de modelos profundos**
* **Viés e ética**

## 7. Recursos e Leituras Complementares

* Jurafsky & Martin, *Speech and Language Processing* (Capítulos sobre ML em PLN)
* Curso de Machine Learning com Python (Scikit-learn)
* Documentação Hugging Face Transformers
* Artigos recentes em conferências ACL, EMNLP e NAACL











## 8. Exercícios Sugeridos

* Implementar um classificador de sentimentos usando Naive Bayes
* Experimentar diferentes vetorizadores (CountVectorizer, TF-IDF)
* Testar embeddings pré-treinados para melhorar a classificação
* Avaliar diferentes métricas para analisar desempenho do modelo


