import pandas as pd
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you have the necessary NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

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
        label = nlp.vocab.strings[match_id]  # get the label
        new_entities.append(Span(doc, start, end, label=label))
    
    # Combine existing and new entities while avoiding overlaps
    non_overlapping_entities = []
    all_entities = list(doc.ents) + new_entities

    # Use a set to track indices of tokens already included
    token_indices = set()
    
    for ent in all_entities:
        if not any(idx in token_indices for idx in range(ent.start, ent.end)):
            non_overlapping_entities.append(ent)
            token_indices.update(range(ent.start, ent.end))
    
    doc.set_ents(non_overlapping_entities, default="unmodified")

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
    pdf_path = 'Data License Agreement.pdf'
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Initialize Matcher for custom entity patterns
    matcher = Matcher(nlp.vocab)
    
    # Add patterns for various entities
    judge_pattern = [{"LOWER": {"IN": ["judge", "magistrate", "justice"]}}]
    matcher.add("JUDGE", [judge_pattern])

    legal_terms = [
        "licensor", "licensee", "agreement", "warranty", "indemnity", "termination", 
        "confidentiality", "liability", "governing law", "jurisdiction", 
        "dispute resolution", "force majeure", "amendment", "clause", "consideration", 
        "default", "entitlement", "notice", "execution", "representations", "assurances"
    ]
    legal_terms_pattern = [{"LOWER": {"IN": legal_terms}}]
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
    
    # Filter for specific entity types
    relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'DATE', 'MONEY', 'PRODUCT', 'LEGAL_TERM', 'VERSION', 'FILETYPE', 'JUDGE']
    
    # Prepare data for CSV
    named_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in relevant_entities]
    print("\nNamed Entities :\n")
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
