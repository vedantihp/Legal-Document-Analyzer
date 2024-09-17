# import pandas as pd
# import pdfplumber
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk import pos_tag
# import spacy
# from spacy.tokens import Span
# from spacy.matcher import Matcher
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Ensure you have the necessary NLTK data files
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')
# # nltk.download('averaged_perceptron_tagger')

# # Path to your PDF file
# pdf_path = 'adobe.pdf'

# # Load spaCy's NER model
# nlp = spacy.load('en_core_web_sm')

# # Initialize Matcher for custom entity patterns
# matcher = Matcher(nlp.vocab)

# # Define patterns for new custom entities
# product_pattern = [{"LOWER": "adobe"}, {"LOWER": "creative"}, {"LOWER": "cloud"}]
# matcher.add("PRODUCT", [product_pattern])

# legal_terms_pattern = [{"LOWER": {"IN": ["licensor", "licensee", "agreement", "warranty"]}}]
# matcher.add("LEGAL_TERM", [legal_terms_pattern])

# version_pattern = [{"TEXT": {"REGEX": r"v\d+\.\d+\.\d+"}}]
# matcher.add("VERSION", [version_pattern])

# filetype_pattern = [{"LOWER": {"IN": ["pdf", "exe", "docx", "xlsx"]}}]
# matcher.add("FILETYPE", [filetype_pattern])

# # Open the PDF file and extract text
# with pdfplumber.open(pdf_path) as pdf:
#     all_text = ""
#     for page in pdf.pages:
#         text = page.extract_text()
#         if text:  # Avoid None types
#             all_text += text

# # Convert text to lowercase for NLTK processing
# all_text_lower = all_text.lower()

# # NLTK Tokenization, Stopword Removal, Stemming, and Lemmatization
# tokens = word_tokenize(all_text_lower)
# stop_words = set(stopwords.words('english'))
# filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# # Stemming
# stemmer = PorterStemmer()
# stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# # Lemmatization
# lemmatizer = WordNetLemmatizer()
# lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# # Reassemble tokens back into text (optional)
# stemmed_text = ' '.join(stemmed_tokens)
# lemmatized_text = ' '.join(lemmatized_tokens)

# # Apply spaCy's NER model on lemmatized text (you can switch to stemmed_text if desired)
# doc = nlp(lemmatized_text)

# # Apply custom matcher to the document
# matches = matcher(doc)

# # Collect custom entities from the matcher
# new_entities = []
# for match_id, start, end in matches:
#     span = doc[start:end]  # get the span of tokens
#     label = nlp.vocab.strings[match_id]  # get the label (PRODUCT, LEGAL_TERM, etc.)
#     new_entities.append(Span(doc, start, end, label=label))  # create a Span object with token indices
#     print(f"Matched span: {span.text}, Label: {label}")  # Debug print

# # Print all recognized entities before adding new ones
# print("Existing entities:")
# print([(ent.text, ent.label_) for ent in doc.ents])  # Debug print

# # Remove existing entities that overlap with new entities
# non_overlapping_entities = []
# for ent in doc.ents:
#     if not any(ent.start <= ne.end and ent.end >= ne.start for ne in new_entities):
#         non_overlapping_entities.append(ent)

# # Add the custom entities to the document's existing entities
# doc.set_ents(non_overlapping_entities + new_entities, default="unmodified")

# # Print all recognized entities after adding new ones
# print("Final entities:")
# print([(ent.text, ent.label_) for ent in doc.ents])  # Debug print

# # Filter for specific entity types (e.g., ORGANIZATION, PERSON, etc.)
# relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'DATE', 'MONEY', 'PRODUCT', 'LEGAL_TERM', 'VERSION', 'FILETYPE']

# # Print Named Entities detected by spaCy after lemmatization
# print("\nNamed Entities (Filtered, Post-Lemmatization):\n")
# for ent in doc.ents:
#     if ent.label_ in relevant_entities:
#         print(f"Entity: {ent.text}, Type: {ent.label_}")

# # Convert stop_words set to list for TfidfVectorizer
# vectorizer = TfidfVectorizer(max_features=10, stop_words=list(stop_words))

# # Fit the vectorizer to the extracted text
# X = vectorizer.fit_transform([all_text])

# # Print top 10 keywords based on TF-IDF scores (after stopword removal)
# print("\nTop Keywords (TF-IDF after stopword removal):\n")
# keywords = vectorizer.get_feature_names_out()
# print(keywords)

# # Save top keywords to a CSV file
# df_keywords = pd.DataFrame(keywords, columns=['Top Keywords'])
# df_keywords.to_csv('top_keywords.csv', index=False)
# print("\nTop Keywords saved to 'top_keywords.csv'")

import pandas as pd
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you have the necessary NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def load_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text
    return all_text

def preprocess_text(text):
    """Preprocess text with NLTK: tokenization, stopword removal, stemming, and lemmatization."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return {
        "stemmed_text": ' '.join(stemmed_tokens),
        "lemmatized_text": ' '.join(lemmatized_tokens),
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "stop_words": stop_words
    }

def perform_ner(text, nlp, matcher):
    """Perform Named Entity Recognition using spaCy and custom entities."""
    doc = nlp(text)
    matches = matcher(doc)
    
    # Collect custom entities from the matcher
    new_entities = []
    for match_id, start, end in matches:
        span = doc[start:end]  # get the span of tokens
        label = nlp.vocab.strings[match_id]  # get the label (PRODUCT, LEGAL_TERM, etc.)
        new_entities.append(Span(doc, start, end, label=label))
    
    # Combine existing and new entities
    non_overlapping_entities = []
    for ent in doc.ents:
        if not any(ent.start <= ne.end and ent.end >= ne.start for ne in new_entities):
            non_overlapping_entities.append(ent)
    
    doc.set_ents(non_overlapping_entities + new_entities, default="unmodified")

    return doc

def extract_keywords(text, stop_words):
    """Extract top keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=10, stop_words=list(stop_words))
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

def save_to_csv(data, filename):
    """Save data to a CSV file."""
    df = pd.DataFrame(data, columns=['Entity', 'Type'])
    df.to_csv(filename, index=False)
    print(f"\nData saved to '{filename}'")

def main():
    pdf_path = 'adobe.pdf'
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Initialize Matcher for custom entity patterns
    matcher = Matcher(nlp.vocab)
    product_pattern = [{"LOWER": "adobe"}, {"LOWER": "creative"}, {"LOWER": "cloud"}]
    matcher.add("PRODUCT", [product_pattern])
    legal_terms_pattern = [{"LOWER": {"IN": ["licensor", "licensee", "agreement", "warranty"]}}]
    matcher.add("LEGAL_TERM", [legal_terms_pattern])
    version_pattern = [{"TEXT": {"REGEX": r"v\d+\.\d+\.\d+"}}]
    matcher.add("VERSION", [version_pattern])
    filetype_pattern = [{"LOWER": {"IN": ["pdf", "exe", "docx", "xlsx"]}}]
    matcher.add("FILETYPE", [filetype_pattern])

    # Load and preprocess text
    all_text = load_pdf_text(pdf_path)
    processed_text = preprocess_text(all_text)
    
    # Perform NER
    doc = perform_ner(processed_text["lemmatized_text"], nlp, matcher)
    
    # Filter for specific entity types (e.g., ORGANIZATION, PERSON, etc.)
    relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'DATE', 'MONEY', 'PRODUCT', 'LEGAL_TERM', 'VERSION', 'FILETYPE']
    
    # Prepare data for CSV
    named_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in relevant_entities]
    print("\nNamed Entities (Filtered, Post-Lemmatization):\n")
    for ent in named_entities:
        print(f"Entity: {ent[0]}, Type: {ent[1]}")

    save_to_csv(named_entities, 'named_entities.csv')

    # Extract and save keywords
    keywords = extract_keywords(all_text, processed_text["stop_words"])
    print("\nTop Keywords (TF-IDF after stopword removal):\n")
    print(keywords)

    df_keywords = pd.DataFrame(keywords, columns=['Top Keywords'])
    df_keywords.to_csv('top_keywords.csv', index=False)
    print("\nTop Keywords saved to 'top_keywords.csv'")

if __name__ == "__main__":
    main()
