# Text Summarization Tool

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PALADUGU VISHNU VARDHAN

*INTERN ID*: CODF32

*DOMAIN*: ARTIFICIAL INTELLIGENCE 

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH KUMAR




A Python-based natural language processing tool that summarizes lengthy articles into concise, meaningful summaries.

## Features

- **Extractive Summarization**: Identifies and extracts the most important sentences from the text using graph-based ranking algorithms.
- **Command-line Interface**: Easy-to-use CLI for processing texts and files.
- **Visualization**: Generate visualizations showing the importance of each sentence in the text.
- **Statistics**: Get detailed statistics about the summarization process.

## How It Works

This tool uses the following NLP techniques:

1. **Text Preprocessing**: Cleans and normalizes the input text.
2. **Sentence Tokenization**: Splits the text into individual sentences.
3. **Word Tokenization**: Breaks sentences into words and removes stopwords.
4. **Sentence Similarity Calculation**: Computes the similarity between sentences using cosine similarity.
5. **Graph-based Ranking**: Uses the PageRank algorithm to rank sentences by importance.
6. **Summary Generation**: Selects the top-ranked sentences to form a coherent summary.

## Installation

```bash
# Clone the repository
git clone https://github.com/Ver-er/Text-Summarization-Tool
cd Text-Summarization-Tool

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Summarize text from a file
python summarize_text.py -f sample_article.txt

# Summarize text directly from command line
python summarize_text.py -t "Your lengthy text goes here."
```

### Advanced Options

```bash
# Specify the number of sentences in the summary
python summarize_text.py -f sample_article.txt -n 3

# Save the summary to a file
python summarize_text.py -f sample_article.txt -o summary.txt

# Generate a visualization of sentence importance
python summarize_text.py -f sample_article.txt -v
```

### As a Module

You can also use the tool as a module in your Python scripts:

```python
from text_summarizer import extractive_summarize

text = "Your lengthy text here..."
summary = extractive_summarize(text, num_sentences=5)
print(summary)
```

## Example

### Input Text

A sample article about artificial intelligence is included in the repository (`sample_article.txt`).

### Generated Summary

```
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. Machine learning, a subset of AI, focuses on the development of computer programs that can access data and use it to learn for themselves. In healthcare, AI applications can help doctors make better diagnoses, develop treatment plans, and even discover new drugs. Looking to the future, the development of artificial general intelligence (AGI) – systems capable of understanding, learning, and applying knowledge across a wide range of tasks – represents one of the most exciting and challenging frontiers in AI research. In conclusion, artificial intelligence represents one of the most transformative technologies of our time.
```

### Visualization

When you run the summarizer with the `-v` flag, it generates a bar chart showing the importance score of each sentence:

![Sentence Importance](sentence_importance_example.png)

## Requirements

- Python 3.6+
- NLTK
- NumPy
- NetworkX
- Matplotlib

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 

