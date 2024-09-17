import pandas as pd
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Path to your PDF file
pdf_path = 'HertzGroupRealtyTrustInc_20190920_S-11A_EX-10.8_11816941_EX-10.8_Trademark License Agreement.pdf'

# Load spaCy's NER model
nlp = spacy.load('en_core_web_sm')

# Open the PDF file and extract text
with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:  # Avoid None types
            all_text += text

# Convert text to lowercase for NLTK processing
all_text_lower = all_text.lower()

# NLTK Tokenization, Stopword Removal, Stemming, and Lemmatization
tokens = word_tokenize(all_text_lower)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Reassemble tokens back into text (optional)
stemmed_text = ' '.join(stemmed_tokens)
lemmatized_text = ' '.join(lemmatized_tokens)

# POS tagging (optional for NLTK)
pos_tagged_tokens = pos_tag(lemmatized_tokens)

# Output the NLTK preprocessed tokens (optional)
print("\nPreprocessed Tokens (Stemming):\n", stemmed_tokens)
print("\nPreprocessed Tokens (Lemmatization):\n", lemmatized_tokens)

# Apply spaCy's NER model on lemmatized text (you can switch to stemmed_text if desired)
doc = nlp(lemmatized_text)

# Filter for specific entity types (e.g., ORGANIZATION, PERSON, etc.)
relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'DATE', 'MONEY']

# Print Named Entities detected by spaCy after lemmatization
print("\nNamed Entities (Filtered, Post-Lemmatization):\n")
for ent in doc.ents:
    if ent.label_ in relevant_entities:
        print(f"Entity: {ent.text}, Type: {ent.label_}")

# Convert stop_words set to list for TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10, stop_words=list(stop_words))

# Fit the vectorizer to the extracted text
X = vectorizer.fit_transform([all_text])

# Print top 10 keywords based on TF-IDF scores (after stopword removal)
print("\nTop Keywords (TF-IDF after stopword removal):\n")
print(vectorizer.get_feature_names_out())

keywords = vectorizer.get_feature_names_out()
df_keywords = pd.DataFrame(keywords, columns=['Top Keywords'])
df_keywords.to_csv('top_keywords.csv', index=False)
print("\nTop Keywords saved to 'top_keywords.csv'")
