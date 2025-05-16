#!/usr/bin/env python3
"""
Text Summarization Tool

This script uses natural language processing techniques to summarize lengthy articles.
It implements extractive summarization by identifying the most important sentences.
"""

import nltk
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def simple_sentence_tokenize(text):
    """A simple sentence tokenizer as a fallback"""
    # Replace common sentence-ending punctuation with a unique marker
    text = text.replace('. ', '.<SPLIT>')
    text = text.replace('! ', '!<SPLIT>')
    text = text.replace('? ', '?<SPLIT>')
    
    # Split by the marker
    sentences = text.split('<SPLIT>')
    
    # Clean up the sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def preprocess_text(text):
    """Clean and preprocess the text."""
    # Remove extra whitespace, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters and digits (except periods for sentence splitting)
    text = re.sub(r'[^\w\s\.]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def sentence_similarity(sent1, sent2, stopwords=None):
    """Compute the similarity between two sentences using cosine similarity."""
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]
    
    # Create word frequency dictionaries
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build vectors
    for w in sent1:
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    # Handle empty vectors
    if sum(vector1) == 0 or sum(vector2) == 0:
        return 0.0
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    """Build a similarity matrix for all sentences."""
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(
                    sentences[i], sentences[j], stop_words)
    
    return similarity_matrix

def extractive_summarize(text, num_sentences=5):
    """
    Generate an extractive summary by ranking sentences using a graph-based algorithm.
    
    Args:
        text (str): The input text to summarize
        num_sentences (int): The number of sentences to include in the summary
        
    Returns:
        str: The summarized text
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize the text into sentences
    try:
        sentences = sent_tokenize(preprocessed_text)
    except Exception:
        # Fallback to simple tokenizer if NLTK tokenizer fails
        sentences = simple_sentence_tokenize(preprocessed_text)
    
    # Return original text if it's too short
    if len(sentences) <= num_sentences:
        return text
    
    # Tokenize words in each sentence
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback to a small set of common English stopwords
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                      "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                      'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                      'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                      'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                      'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                      'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                      'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                      'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                      'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                      'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])
    
    # Simple word tokenization fallback
    sentence_tokens = []
    for s in sentences:
        words = []
        for word in s.lower().split():
            word = word.strip(string.punctuation)
            if word:
                words.append(word)
        sentence_tokens.append(words)
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(sentence_tokens, stop_words)
    
    # Rank sentences using PageRank algorithm
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # Sort sentences by score and select top N
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select top 'num_sentences' sentences
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
    
    # Join the top sentences to form the summary
    summary = ' '.join([s[2] for s in top_sentences])
    
    return summary

def display_summary_statistics(original_text, summary):
    """
    Display statistics about the summarization process.
    
    Args:
        original_text (str): The original text
        summary (str): The generated summary
    """
    original_word_count = len(original_text.split())
    summary_word_count = len(summary.split())
    
    try:
        original_sentence_count = len(sent_tokenize(original_text))
    except Exception:
        original_sentence_count = len(simple_sentence_tokenize(original_text))
        
    try:
        summary_sentence_count = len(sent_tokenize(summary))
    except Exception:
        summary_sentence_count = len(simple_sentence_tokenize(summary))
    
    reduction_percentage = ((original_word_count - summary_word_count) / original_word_count) * 100
    
    print("\nSummarization Statistics:")
    print(f"Original Word Count: {original_word_count}")
    print(f"Summary Word Count: {summary_word_count}")
    print(f"Original Sentence Count: {original_sentence_count}")
    print(f"Summary Sentence Count: {summary_sentence_count}")
    print(f"Reduction: {reduction_percentage:.2f}%")

def visualize_sentence_importance(text, num_top_sentences=5):
    """
    Create a visual representation of sentence importance scores.
    
    Args:
        text (str): The input text
        num_top_sentences (int): Number of top sentences to highlight
    """
    # Preprocess and tokenize text
    preprocessed_text = preprocess_text(text)
    try:
        sentences = sent_tokenize(preprocessed_text)
    except Exception:
        sentences = simple_sentence_tokenize(preprocessed_text)
    
    if len(sentences) <= 1:
        print("Text is too short to visualize sentence importance.")
        return
    
    # Tokenize words in each sentence
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                     "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                     'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                     'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                     'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                     'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                     'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                     'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                     'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                     'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                     'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])
    
    # Simple word tokenization
    sentence_tokens = []
    for s in sentences:
        words = []
        for word in s.lower().split():
            word = word.strip(string.punctuation)
            if word:
                words.append(word)
        sentence_tokens.append(words)
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(sentence_tokens, stop_words)
    
    # Rank sentences using PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # Prepare data for visualization
    sentence_indices = list(range(len(sentences)))
    importance_scores = [scores[i] for i in sentence_indices]
    
    # Sort by importance and get top sentences
    sorted_indices = sorted(range(len(importance_scores)), key=lambda i: importance_scores[i], reverse=True)
    top_indices = sorted_indices[:num_top_sentences]
    
    # Set up colors - highlight top sentences
    colors = ['lightgrey'] * len(sentences)
    for idx in top_indices:
        colors[idx] = 'lightblue'
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(sentence_indices, importance_scores, color=colors)
    plt.xlabel('Sentence Index')
    plt.ylabel('Importance Score')
    plt.title('Sentence Importance in Text')
    plt.xticks(sentence_indices)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sentence_importance.png')
    plt.close()
    print("Sentence importance visualization saved as 'sentence_importance.png'")

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

    Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, though at the time that was not articulated as a problem separate from artificial intelligence.

    The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it is confronted with. In 2015, Google developed a system capable of learning how to play Atari video games just by watching them. Deep learning-based models have also improved performance across machine translation and parsing.

    Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm of machine learning is different from that of most prior attempts at language processing. Prior implementations of language-processing tasks typically involved the direct hand coding of large sets of rules. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples.
    """
    
    print("=== TEXT SUMMARIZATION TOOL ===")
    print("\nOriginal Text:")
    print(sample_text)
    
    # Generate and display summary
    summary = extractive_summarize(sample_text)
    print("\nSummary:")
    print(summary)
    
    # Display statistics about the summarization
    display_summary_statistics(sample_text, summary)
    
    # Visualize sentence importance
    visualize_sentence_importance(sample_text) 