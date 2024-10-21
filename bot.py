import streamlit as st
import pandas as pd
import pdfplumber
import numpy as np
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained word embeddings (GloVe)
@st.cache_resource
def load_glove_embeddings():
    word_embeddings = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings

word_embeddings = load_glove_embeddings()

# Class for PDF handling
class PDFHandler:
    def read_pdf(self, pdf_file):
        all_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text
        return all_text

pdf_handler = PDFHandler()

# Clean and preprocess the text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove all non-alphabetical characters
    return text.lower()

# Function to remove stopwords
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in sentence.split() if word not in stop_words])

# Custom stopwords to be included in the keyword extraction step
custom_stopwords = set(stopwords.words('english')).union({"agreement", "shall", "use", "data", "database", "licensor", "licensee", "legal"})


# Summarization using word embeddings, cosine similarity, and PageRank
def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)  # Split the text into sentences
    
    # Clean each sentence
    clean_sentences = [clean_text(sentence) for sentence in sentences]
    
    # Remove stopwords
    clean_sentences = [remove_stopwords(sentence) for sentence in clean_sentences]

    # Create sentence vectors using word embeddings
    sentence_vectors = []
    for sentence in clean_sentences:
        if sentence:
            vector = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence.split()]) / (len(sentence.split()) + 0.001)
        else:
            vector = np.zeros((100,))
        sentence_vectors.append(vector)

    # Create a similarity matrix between sentences
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    # Apply PageRank to rank the sentences
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences based on scores
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Return the top 'num_sentences' ranked sentences as the summary
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Text preprocessing with NLTK
def preprocess_text(text):
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

# Perform Named Entity Recognition
def perform_ner(text, nlp, matcher):
    doc = nlp(text)
    matches = matcher(doc)

    new_entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        new_entities.append(Span(doc, start, end, label=label))

    # Combine existing and new entities
    non_overlapping_entities = []
    all_entities = list(doc.ents) + new_entities
    token_indices = set()

    for ent in all_entities:
        if not any(idx in token_indices for idx in range(ent.start, ent.end)):
            non_overlapping_entities.append(ent)
            token_indices.update(range(ent.start, ent.end))

    doc.set_ents(non_overlapping_entities, default="unmodified")
    return doc

# Refined extract_keywords function
def extract_keywords(text, stop_words, num_keywords=10, use_bigrams=True):
    # Initialize vectorizer with custom stop words
    if use_bigrams:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words=list(stop_words), ngram_range=(1, 2))  # Bigram/Unigram
    else:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words=list(stop_words))  # Unigram only

    X = vectorizer.fit_transform([text])
    
    # Extract keywords with their TF-IDF scores
    keywords = vectorizer.get_feature_names_out()
    return keywords


# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Custom entity patterns
matcher = Matcher(nlp.vocab)
matcher.add("LEGAL_TERM", [[{"LOWER": {"IN": ["licensor", "licensee", "agreement", "warranty", "liability"]}}]])
matcher.add("VERSION", [[{"TEXT": {"REGEX": r"v\d+\.\d+"}}]])
matcher.add("FILETYPE", [[{"LOWER": {"IN": ["pdf", "exe", "docx"]}}]])

# Streamlit UI
st.title('Legal Document Analyzer')

# Sidebar
st.sidebar.title("Menu")
menu = ["Summarize Document", "Show Named Entities", "Extract Top Keywords"]
choice = st.sidebar.selectbox("Choose an option", menu)

# PDF upload
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    text = pdf_handler.read_pdf(uploaded_file)
    
    if choice == "Summarize Document":
        st.subheader("Summarize Document")
        num_sentences = st.slider("Number of sentences in summary", 1, 10, 3)
        if st.button("Summarize"):
            summary = summarize_text(text, num_sentences=num_sentences)
            st.write("Summary:")
            st.write(summary)

    elif choice == "Show Named Entities":
        st.subheader("Show Named Entities")
        if st.button("Show Entities"):
            processed_text = preprocess_text(text)
            doc = perform_ner(processed_text["lemmatized_text"], nlp, matcher)

            # Extract named entities and store in a list
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Convert named entities to DataFrame for tabular display
            df_entities = pd.DataFrame(named_entities, columns=['Entity', 'Type'])
            st.write("Named Entities:")
            st.table(df_entities)

    elif choice == "Extract Top Keywords":
        st.subheader("Extract Top Keywords")
        if st.button("Extract Keywords"):
            processed_text = preprocess_text(text)
            keywords = extract_keywords(text, processed_text["stop_words"])
            st.write("Top Keywords:")
            for keyword in keywords:
                st.write(f"- {keyword}")
