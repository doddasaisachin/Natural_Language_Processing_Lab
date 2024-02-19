import spacy 
nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing is a fascinating field of study." 
doc = nlp(text) 
tokens = [token.text for token in doc] 
lemmas = [token.lemma_ for token in doc] 
print("Tokens:", tokens) 
print("Lemmas:", lemmas) 
print("\nDependency Parsing:")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])