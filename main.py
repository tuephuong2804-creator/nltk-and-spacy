# ===== IMPORT =====
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===== DOWNLOAD NLTK =====
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ===== LOAD SPACY =====
nlp = spacy.load("en_core_web_sm")

# ===== INPUT TEXT =====
texts = [
    "I love learning natural language processing.",
    "I do not like waking up early.",
    "Are you going to the party tonight?"
]

# ===== NLTK SETUP =====
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("===== NLTK PROCESSING =====")

nltk_results = []

for text in texts:
    print("\nOriginal:", text)

    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]

    stems = [stemmer.stem(w) for w in tokens]
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]

    nltk_results.append(lemmas)

    print("Tokens:", tokens)
    print("Stems:", stems)
    print("Lemmas:", lemmas)


print("\n===== SPACY PROCESSING =====")

spacy_results = []

for text in texts:
    print("\nOriginal:", text)

    doc = nlp(text)

    tokens = []
    lemmas = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        tokens.append(token.text.lower())
        lemmas.append(token.lemma_.lower())

    spacy_results.append(lemmas)

    print("Tokens:", tokens)
    print("Lemmas:", lemmas)


# ===== GOLD DATA (FIXED - MATCH INPUT) =====
gold = [
    ['love', 'learn', 'natural', 'language', 'processing'],
    ['like', 'wake', 'early'],
    ['go', 'party', 'tonight']
]


# ===== TOKEN-LEVEL EVALUATION (GOOD VERSION) =====
def compare_token_level(pred, gold):
    y_true = []
    y_pred = []

    for p, g in zip(pred, gold):
        for word in g:
            y_true.append(1)
            y_pred.append(1 if word in p else 0)

    return y_true, y_pred


# ===== EVALUATE =====
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, zero_division=0))


# NLTK
y_true_nltk, y_pred_nltk = compare_token_level(nltk_results, gold)
evaluate("NLTK", y_true_nltk, y_pred_nltk)

# spaCy
y_true_spacy, y_pred_spacy = compare_token_level(spacy_results, gold)
evaluate("spaCy", y_true_spacy, y_pred_spacy)