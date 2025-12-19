import os
import json
import numpy as np
from bert_score import score
from ner_anonymizer import NERAnonymizer
import pandas as pd
from tqdm import tqdm


def read_text_file(file_path):
    """Read text from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def calculate_bert_score(original_texts, anonymized_texts, model_type="bert-base-multilingual-cased"):
    """
    Calculate BERTScore between original and anonymized texts
    
    Args:
        original_texts: List of original texts
        anonymized_texts: List of anonymized texts
        model_type: BERT model to use for scoring
        
    Returns:
        Tuple of (precision, recall, f1) lists
    """
    P, R, F1 = score(
        anonymized_texts, 
        original_texts, 
        lang="ru", 
        model_type=model_type,
        verbose=False
    )
    
    return P.numpy(), R.numpy(), F1.numpy()


def process_eval_directory(eval_dir="eval", output_file="bert_scores.csv"):
    """
    Process all text files in the eval directory and calculate BERTScore
    
    Args:
        eval_dir: Directory containing test files
        output_file: CSV file to save results
    """
    # Initialize anonymizers
    mask_anonymizer = NERAnonymizer(mode="mask")
    replace_anonymizer = NERAnonymizer(mode="replace")
    
    # Get all text files in the eval directory
    file_names = [f for f in os.listdir(eval_dir) if f.endswith('.txt')]
    file_names.sort()  # Sort to ensure consistent ordering
    
    print(f"Found {len(file_names)} files to process")
    
    # Lists to store results
    results = []
    
    # Process each file
    for file_name in tqdm(file_names, desc="Processing files"):
        file_path = os.path.join(eval_dir, file_name)
        original_text = read_text_file(file_path)
        
        # Apply both anonymization methods
        masked_text, _ = mask_anonymizer.extract_and_anonymize(original_text)
        replaced_text, _ = replace_anonymizer.extract_and_anonymize(original_text)
        
        # Calculate BERTScore for masked text
        mask_precision, mask_recall, mask_f1 = calculate_bert_score(
            [original_text], [masked_text]
        )
        
        # Calculate BERTScore for replaced text
        replace_precision, replace_recall, replace_f1 = calculate_bert_score(
            [original_text], [replaced_text]
        )
        
        # Store results
        result = {
            'file_name': file_name,
            'mask_precision': mask_precision[0],
            'mask_recall': mask_recall[0],
            'mask_f1': mask_f1[0],
            'replace_precision': replace_precision[0],
            'replace_recall': replace_recall[0],
            'replace_f1': replace_f1[0]
        }
        results.append(result)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Calculate statistics
    calculate_statistics(df)
    
    return df


def calculate_statistics(df):
    """
    Calculate and print statistics for the BERTScore results
    
    Args:
        df: DataFrame with BERTScore results
    """
    print("\n=== BERTScore Statistics ===")
    
    # Mask method statistics
    print("\nMask Method:")
    print(f"Mean Precision: {df['mask_precision'].mean():.4f}")
    print(f"Mean Recall: {df['mask_recall'].mean():.4f}")
    print(f"Mean F1: {df['mask_f1'].mean():.4f}")
    print(f"Median F1: {df['mask_f1'].median():.4f}")
    print(f"25th percentile F1: {df['mask_f1'].quantile(0.25):.4f}")
    print(f"75th percentile F1: {df['mask_f1'].quantile(0.75):.4f}")
    
    # Replace method statistics
    print("\nReplace Method:")
    print(f"Mean Precision: {df['replace_precision'].mean():.4f}")
    print(f"Mean Recall: {df['replace_recall'].mean():.4f}")
    print(f"Mean F1: {df['replace_f1'].mean():.4f}")
    print(f"Median F1: {df['replace_f1'].median():.4f}")
    print(f"25th percentile F1: {df['replace_f1'].quantile(0.25):.4f}")
    print(f"75th percentile F1: {df['replace_f1'].quantile(0.75):.4f}")
    
    # Comparison
    print("\n=== Comparison ===")
    print(f"Mean F1 difference (Replace - Mask): {df['replace_f1'].mean() - df['mask_f1'].mean():.4f}")
    print(f"Median F1 difference (Replace - Mask): {df['replace_f1'].median() - df['mask_f1'].median():.4f}")
    
    # Count which method performs better for each file
    replace_better_count = (df['replace_f1'] > df['mask_f1']).sum()
    mask_better_count = (df['mask_f1'] > df['replace_f1']).sum()
    equal_count = (df['replace_f1'] == df['mask_f1']).sum()
    
    print(f"\nFiles where Replace method is better: {replace_better_count} ({replace_better_count/len(df)*100:.1f}%)")
    print(f"Files where Mask method is better: {mask_better_count} ({mask_better_count/len(df)*100:.1f}%)")
    print(f"Files with equal F1 scores: {equal_count} ({equal_count/len(df)*100:.1f}%)")


def main():
    """Main function to run the evaluation"""
    print("Starting BERTScore evaluation of anonymization methods...")
    
    # Process all files and calculate BERTScore
    results_df = process_eval_directory()
    
    print("\nEvaluation completed!")
    print("Results saved to bert_scores.csv")
    print("\nConclusion:")
    print("Higher BERTScore indicates better semantic similarity to the original text.")
    print("Compare the F1 scores to determine which anonymization method preserves")
    print("more of the original meaning while protecting personal data.")


if __name__ == "__main__":
    main()