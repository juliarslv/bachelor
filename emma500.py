from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import re
import json

# Model
model_name = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Languages to process
languages = ["en", "de", "ar", "ru", "be", "mn", "zh-classical"]  # English, German, Arabic, Russian, Belarusian, Mongolian, Chinese

# Splitting a text into sentences using a regex-based approach.
def split_into_sentences(text):
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(\s|\n)'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Pre-Tokanization: Grapheme Pair Encoding
def apply_gpe(text):
    return " ".join([text[i:i+2] for i in range(0, len(text), 2)])

# Tokenization Function
def tokenize(texts, tokenizer):
    tokenized_texts = [tokenizer.tokenize(text) for text in texts] # List of tokenized texts.
    token_counts = [len(tokens) for tokens in tokenized_texts] # List of token counts for each text.
    return {
        "tokenized_texts": tokenized_texts,
        "token_counts": token_counts
    }

# Metric Calculation Functions
# Compression Ratio = Number of Characters / Number of Tokens
def calculate_max_compression_ratio(texts, token_counts):
    compression_ratios = [
        len(text) / token_count if token_count > 0 else 0
        for text, token_count in zip(texts, token_counts)
    ]
    return max(compression_ratios) if compression_ratios else 0

# Min Tokenization Parity = Minimum Token Count / Maximum Token Count
def calculate_min_tokenization_parity(token_counts):
    if not token_counts:
        return 0
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    return min_tokens / max_tokens if max_tokens > 0 else 0

# Main Processing Function
# Process a single language: tokenize with and without GPE, calculate metrics, and return results.
def process_language(lang, tokenizer):
    print(f"\nProcessing dataset for language: {lang}")
    try:
        # Loading the dataset and extract the text
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
        documents = dataset_lang["text"][:1000]
        
        # Sentence splitting
        sentences = []
        for doc in documents:
            sentences.extend(split_into_sentences(doc))
        sentences = sentences[:1000]

        # Tokenizing without GPE
        tokenized_data_without_gpe = tokenize(sentences, tokenizer)

        # Tokenizing with GPE
        gpe_sentences = [apply_gpe(sentence) for sentence in sentences]
        tokenized_data_with_gpe = tokenize(gpe_sentences, tokenizer)

        # Displaying the first sentence and tokenized forms
        print(f"\nLanguage: {lang}")
        print(f"Original Sentence: {sentences[0]}")
        print(f"Tokenized Sentence (No GPE): {tokenized_data_without_gpe['tokenized_texts'][0]}")
        print(f"Tokenized Sentence (With GPE): {tokenized_data_with_gpe['tokenized_texts'][0]}")

        # Computing metrics
        metrics = {
            "Max Compression (No GPE)": calculate_max_compression_ratio(sentences, tokenized_data_without_gpe["token_counts"]),
            "Min Parity (No GPE)": calculate_min_tokenization_parity(tokenized_data_without_gpe["token_counts"]),
            "Max Compression (With GPE)": calculate_max_compression_ratio(gpe_sentences, tokenized_data_with_gpe["token_counts"]),
            "Min Parity (With GPE)": calculate_min_tokenization_parity(tokenized_data_with_gpe["token_counts"]),
        }

        # Displaying metrics
        print("\nMetrics for language:", lang)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")

        # Storing results
        result = {
            "Language": lang,
            **metrics,
        }

        return result

    except Exception as e:
        print(f"Error processing {lang}: {e}")
        return None

# Execution for All Languages
results = []

for lang in languages:
    result = process_language(lang, tokenizer)
    if result:
        results.append(result)

# Saving results to JSON
results_json = {"results": results}
with open("computations_emma500.json", "w") as json_file:
    json.dump(results_json, json_file, indent=4)
print("\nResults saved to computations_emma500.json")
