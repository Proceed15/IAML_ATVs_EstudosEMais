## 4. Exemplo com spaCy (Árvore de Dependência)

```python
import spacy
from spacy import displacy

# Carrega modelo para português
nlp = spacy.load("pt_core_news_sm")

frase = nlp("João gosta de sorvete")

for token in frase:
    print(f"{token.text} → {token.head.text} ({token.dep_})")
```

### Visualização da árvore

```python
displacy.render(frase, style='dep', jupyter=True)
```

## 5. Exemplo com NLTK (Constituency Parsing)

```python
import nltk
from nltk import CFG

gramatica = CFG.fromstring("""
S -> NP VP
NP -> 'João'
VP -> V NP | V PP
V -> 'gosta'
PP -> P NP
P -> 'de'
NP -> 'sorvete'
""")

parser = nltk.ChartParser(gramatica)

for tree in parser.parse(['João', 'gosta', 'de', 'sorvete']):
    tree.pretty_print()
```

> Nota: o NLTK exige definições explícitas de gramáticas e sentenças, sendo mais indicado para fins didáticos.

5. Exemplo com NLTK (Constituency Parsing)
import nltk
from nltk import CFG

gramatica = CFG.fromstring("""
S -> NP VP
NP -> 'João'
VP -> V NP | V PP
V -> 'gosta'
PP -> P NP
P -> 'de'
NP -> 'sorvete'
""")


parser = nltk.ChartParser(gramatica)
for tree in parser.parse(['João', 'gosta', 'de', 'sorvete']):
    tree.pretty_print()
Nota: o NLTK exige definições explícitas de gramáticas e sentenças, sendo mais indicado para fins didáticos.




