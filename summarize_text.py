#!/usr/bin/env python3
"""
Text Summarization CLI Tool

A command-line interface for the text summarization functionality.
"""

import argparse
import sys
from text_summarizer import extractive_summarize, display_summary_statistics, visualize_sentence_importance

def read_text_from_file(file_path):
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def save_summary_to_file(summary, output_file):
    """Save the summary to a file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(summary)
        print(f"Summary saved to {output_file}")
    except Exception as e:
        print(f"Error saving summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="Text Summarization Tool")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-t', '--text', help="Text to summarize")
    input_group.add_argument('-f', '--file', help="File containing text to summarize")
    
    # Output options
    parser.add_argument('-o', '--output', help="Output file to save the summary")
    parser.add_argument('-n', '--num-sentences', type=int, default=5, 
                        help="Number of sentences in the summary (default: 5)")
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help="Generate a visualization of sentence importance")
    
    args = parser.parse_args()
    
    # Get the input text
    if args.text:
        text = args.text
    else:
        text = read_text_from_file(args.file)
    
    # Generate summary
    summary = extractive_summarize(text, args.num_sentences)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(summary)
    
    # Display statistics
    display_summary_statistics(text, summary)
    
    # Save to file if specified
    if args.output:
        save_summary_to_file(summary, args.output)
    
    # Generate visualization if requested
    if args.visualize:
        visualize_sentence_importance(text, args.num_sentences)

if __name__ == "__main__":
    main() 