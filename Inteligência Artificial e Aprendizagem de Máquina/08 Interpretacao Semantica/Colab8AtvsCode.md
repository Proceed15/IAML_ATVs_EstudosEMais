### Exemplo Prático com spaCy

```python
!python -m spacy download pt_core_news_sm
!python -m spacy download pt_core_news_md

import spacy

nlp = spacy.load("pt_core_news_sm")
doc = nlp("O gato subiu no telhado.")

for token in doc:
    print(f"{token.text:<12} | {token.dep_:<12} | {token.head.text}")
```

Saída esperada:

```
O            | det          | gato
gato         | nsubj        | subiu
subiu        | ROOT         | subiu
no           | case         | telhado
telhado      | obl          | subiu
.            | punct        | subiu
```

Com isso, podemos inferir que o **sujeito** da ação é “gato” e o **objeto indireto** é “telhado”.

---

### Exemplo Prático com NLTK

```python
import nltk
from nltk import CFG

grammar = CFG.fromstring("""
  S -> NP VP
  NP -> Det N
  VP -> V NP
  Det -> 'o'
  N -> 'menino' | 'livro'
  V -> 'leu'
""")

sentenca = ['o', 'menino', 'leu', 'o', 'livro']
parser = nltk.ChartParser(grammar)

for arvore in parser.parse(sentenca):
    arvore.pretty_print()
```

---

#### 2. Representações Distribuídas (Vetoriais)

Palavras e frases podem ser representadas por **vetores de números** que capturam seu significado com base no **contexto** de uso. Isso permite medir similaridades semânticas entre palavras.

**Exemplo com spaCy:**

```python
import spacy

nlp = spacy.load("pt_core_news_md")  # usa vetores de palavra
doc1 = nlp("rei")
doc2 = nlp("rainha")

similaridade = doc1.similarity(doc2)
print(f"Similaridade semântica: {similaridade:.2f}")
```

A saída mostra quão semanticamente próximas são essas palavras. Modelos maiores (`md`, `lg`) têm embeddings treinados para capturar semântica contextual.

#### 3. Ontologias e Grafos Semânticos

Ontologias organizam o conhecimento em hierarquias de conceitos e relações. Por exemplo:

* "Cão" é um tipo de "mamífero"
* "Mamífero" é um tipo de "animal"

Essas relações podem ser representadas em **grafos semânticos**, que conectam conceitos por suas propriedades.

**Ferramentas comuns:**

* WordNet (disponível via NLTK)
* spaCy (para extração de relações)

**Exemplo com WordNet (em inglês):**

```python
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

cachorro = wn.synsets('dog')[0]
print(cachorro.definition())
print(cachorro.hypernyms())  # Conceitos mais gerais
```

---

### Comparação entre Formas de Representação

| Tipo                 | Vantagens                                  | Limitações                              |
| -------------------- | ------------------------------------------ | --------------------------------------- |
| Lógica de predicados | Alta precisão, usada em inferência lógica  | Difícil de aplicar em linguagem ambígua |
| Vetores (embeddings) | Escalável, aplicável em ML e deep learning | Menos interpretável                     |
| Ontologias/Grafos    | Organiza conhecimento de forma estruturada | Requer curadoria manual ou boa extração |

---
### Exemplo Prático: Construção de Gramática e Parser com `nltk.grammar` e `nltk.parse`

```python
import nltk
from nltk.sem import Valuation, Model, Assignment
from nltk import CFG, FeatureChartParser
nltk.download('book_grammars')

# Definindo uma gramática com semântica simbólica
grammar = nltk.load('grammars/book_grammars/simple-sem.fcfg')

# Frase de entrada
sentence = 'Angus sees a dog'.split()

# Parser com suporte a semântica
parser = FeatureChartParser(grammar)

# Parsing + semântica
for tree in parser.parse(sentence):
    print(tree.label()['SEM'])  # Expressão semântica
```

> A gramática `simple-sem.fcfg` vem com o NLTK e define regras como:

```
S[SEM=<?subj(?vp)>] -> NP[SEM=?subj] VP[SEM=?vp]
VP[SEM=<?v(?obj)>] -> V[SEM=?v] NP[SEM=?obj]
NP[SEM=<angus>] -> 'Angus'
NP[SEM=<dog>] -> 'a' 'dog'
V[SEM=<see>] -> 'sees'
```

Essa estrutura resulta na semântica:

```python
see(angus, dog)
```

---

Exemplo:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat chased the mouse.")

for token in doc:
    print(f"{token.text:<10} {token.dep_:<10} {token.head.text:<10} {token.pos_}")
```

Saída:

```
The        det        cat        DET
cat        nsubj      chased     NOUN
chased     ROOT       chased     VERB
the        det        mouse      DET
mouse      dobj       chased     NOUN
.          punct      chased     PUNCT
```

---

### Uso de `Doc`, `Token` e `Span` para Inferência Semântica

* `Doc`: objeto que representa o texto completo.
* `Token`: cada palavra ou símbolo pontuacional.
* `Span`: sequência de tokens (por exemplo, uma entidade).

```python
subject = [tok for tok in doc if tok.dep_ == "nsubj"][0]
verb = doc[0].root
obj = [tok for tok in doc if tok.dep_ == "dobj"][0]

print(f"Ação: {verb.lemma_}({subject.text}, {obj.text})")
```

Resultado:

```
Ação: chase(cat, mouse)
```

---

### Exemplo Prático: Identificar Sujeito, Verbo e Objeto

```python
def analisar_sentenca(texto):
    doc = nlp(texto)
    sujeito = next((tok for tok in doc if tok.dep_ == "nsubj"), None)
    verbo = doc[:].root
    objeto = next((tok for tok in doc if tok.dep_ == "dobj"), None)

    if sujeito and verbo and objeto:
        print(f"{verbo.lemma_}({sujeito.text}, {objeto.text})")
    else:
        print("Não foi possível identificar todos os elementos.")

analisar_sentenca("Maria comprou um livro.")
```

Saída esperada:

```
comprar(Maria, livro)
```

---
### 6.1. Extração de Relações entre Entidades (Relation Extraction)

**Objetivo**: Identificar relações semânticas entre entidades mencionadas em um texto.

**Exemplo**:
Frase: *"Marie Curie discovered radium in Paris."*

Com spaCy:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Marie Curie discovered radium in Paris.")

for ent in doc.ents:
    print(ent.text, ent.label_)

for token in doc:
    if token.dep_ == "nsubj":
        subject = token.text
    if token.dep_ == "dobj":
        obj = token.text
    if token.dep_ == "ROOT":
        action = token.lemma_

print(f"Relação: {action}({subject}, {obj})")
```

Saída esperada:

```
Relação: discover(Marie Curie, radium)
```

Esse padrão pode alimentar bases de conhecimento ou sistemas de resposta a perguntas.

---

### 6.3. Interpretação de Comandos em Linguagem Natural

**Objetivo**: Traduzir instruções verbais em ações computacionais.

**Exemplo**:
Comando: *"Ligue as luzes da cozinha."*

Aplicação:

* Identifica-se o **verbo** principal: "ligar"
* Identifica-se o **objeto** ou destino da ação: "luzes da cozinha"
* Traduz-se para uma chamada de API:

  ```json
  {
    "acao": "ligar",
    "dispositivo": "luz",
    "local": "cozinha"
  }
  ```

**Implementação base com spaCy**:

```python
doc = nlp("Ligue as luzes da cozinha")

for token in doc:
    print(token.text, token.dep_, token.head.text)

# Interpretação possível:
# ligar(luzes, cozinha)
```

---
### 7.1. Construção de Gramática Simples no NLTK

Neste exercício, vamos construir uma gramática simples usando o **NLTK** para análise sintática de frases.

#### Passos:

1. Criar uma gramática baseada em regras.
2. Analisar uma frase com o parser do NLTK.

#### Script no Colab:

```python
# Importando bibliotecas
import nltk
from nltk import CFG

# Definindo uma gramática simples
grammar = CFG.fromstring("""
  S -> NP VP
  VP -> V NP
  NP -> Det N
  Det -> 'o' | 'a'
  N -> 'homem' | 'mulher'
  V -> 'viu'
""")

# Criação do parser
parser = nltk.ChartParser(grammar)

# Frase para analisar
sentence = ['o', 'homem', 'viu', 'a', 'mulher']

# Analisando a frase
for tree in parser.parse(sentence):
    tree.pretty_print()
```



### 7.2. Extração de Relações e Estrutura Frasal com spaCy

Neste exercício, vamos usar **spaCy** para identificar relações e dependências entre palavras em uma frase.

#### Passos:

1. Instalar o spaCy e carregar o modelo de linguagem.
2. Analisar uma frase e extrair dependências gramaticais.

#### Script no Colab:

```python
# Importando spaCy
import spacy

# Carregando o modelo de linguagem
nlp = spacy.load("pt_core_news_sm")

# Frase para analisar
doc = nlp("Maria comprou um livro na livraria.")

# Mostrando as dependências e palavras
for token in doc:
    print(f"Palavra: {token.text} | Dependência: {token.dep_} | Cabeça: {token.head.text}")
```

Esse script usa o **spaCy** para identificar relações de dependência entre as palavras da frase "Maria comprou um livro na livraria". Ele imprime as palavras, suas dependências e as palavras principais (head) com as quais elas se conectam.

---


