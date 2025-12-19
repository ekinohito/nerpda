import json

def calculate_mean_f1(results_file):
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    error_books = []
    
    for file_result in data['file_results']:
        metrics = file_result['metrics']
        false_positives = metrics['false_positives']
        false_negatives = metrics['false_negatives']
        f1 = metrics['f1']
        
        # Check if the book has at least one false positive or false negative
        if f1 > 0 or false_negatives > 0 or false_positives > 0:
            error_books.append({
                'file': file_result['file'],
                'f1': f1,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
    
    # Calculate mean F1 score for books with errors
    if error_books:
        mean_f1 = sum(book['f1'] for book in error_books) / len(error_books)
    else:
        mean_f1 = 0.0
    
    return mean_f1, len(error_books), error_books

if __name__ == "__main__":
    results_file = 'natasha_ner_evaluation_results.json'
    mean_f1, count, error_books = calculate_mean_f1(results_file)
    
    print(f"Number of books: {count}")
    print(f"Mean F1 score for books : {mean_f1:.4f}")
    
    # Print some details
    print("\nBooks (sorted by F1 score):")
    for book in sorted(error_books, key=lambda x: x['f1'], reverse=True):
        print(f"{book['file']}: F1={book['f1']:.4f}, FP={book['false_positives']}, FN={book['false_negatives']}")