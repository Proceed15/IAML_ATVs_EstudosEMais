## 4. Exemplo de Intenções

```json
{
  "intencoes": [
    {
      "tag": "saudacao",
      "padroes": ["oi", "olá", "bom dia", "boa tarde"],
      "respostas": ["Olá, como posso ajudar?", "Oi! Em que posso ajudar hoje?"]
    },
    {
      "tag": "despedida",
      "padroes": ["tchau", "até logo", "adeus"],
      "respostas": ["Até logo!", "Tchau, tenha um bom dia!"]
    },
    {
      "tag": "pedido_cardapio",
      "padroes": ["quero ver o cardápio", "tem algo para comer?", "o que vocês servem?"],
      "respostas": ["Temos hambúrgueres, batatas fritas e refrigerantes."]
    }
  ]
}
```


## 5. Implementação com Scikit-learn

### 5.1 Pré-processamento e dados

```python
intencoes = {
    "saudacao": ["oi", "olá", "bom dia", "boa tarde"],
    "despedida": ["tchau", "até logo", "adeus"],
    "pedido_cardapio": ["quero ver o cardápio", "tem algo para comer?", "o que vocês servem?"]
}

frases = []
tags = []

for tag, padroes in intencoes.items():
    for frase in padroes:
        frases.append(frase)
        tags.append(tag)
```

### 5.2 Treinamento do Classificador

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vetor = CountVectorizer()
X = vetor.fit_transform(frases)

modelo = MultinomialNB()
modelo.fit(X, tags)
```

### 5.3 Função de resposta

```python
def responder(frase):
    X_input = vetor.transform([frase])
    tag_prevista = modelo.predict(X_input)[0]

    respostas = {
        "saudacao": ["Olá! Como posso ajudar?", "Oi! Tudo bem?"],
        "despedida": ["Tchau!", "Volte sempre!"],
        "pedido_cardapio": ["Temos hambúrgueres, refrigerantes e sobremesas."]
    }

    return respostas.get(tag_prevista, ["Desculpe, não entendi."])[0]

# Teste
print(responder("oi"))
print(responder("tem algo para comer?"))
```
