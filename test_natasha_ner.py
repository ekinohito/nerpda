import os
import re
from collections import defaultdict
from ner import NatashaNER


class FactRuEvalTester:
    def __init__(self, dataset_path="factRuEval-2016-master/devset"):
        self.dataset_path = dataset_path
        self.ner = NatashaNER()
        
    def parse_spans_file(self, spans_file_path):
        """
        Parse a .spans file and extract name and surname entities.
        
        Returns a list of tuples: (start_pos, end_pos, entity_type)
        where entity_type is either 'NAME' or 'SURNAME'
        """
        entities = []
        
        with open(spans_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                    
                entity_type = parts[1]
                if entity_type not in ['name', 'surname']:
                    continue
                    
                # Extract start and end positions
                start_pos = int(parts[2])
                length = int(parts[3])
                end_pos = start_pos + length
                
                # Convert to our format
                normalized_type = 'NAME' if entity_type == 'name' else 'SURNAME'
                entities.append((start_pos, end_pos, normalized_type))
                
        return entities
    
    def read_text_file(self, txt_file_path):
        """Read the original text from a .txt file."""
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_entities_from_ner(self, text):
        """
        Run NatashaNER on text and extract name and surname entities.
        
        Returns a list of tuples: (start_pos, end_pos, entity_type)
        """
        ner_results = self.ner.extract_names(text)
        entities = []
        
        for result in ner_results:
            entity_type = result[0]  # 'NAME' or 'SURNAME'
            start_pos = result[2]
            end_pos = result[3]
            entities.append((start_pos, end_pos, entity_type))
            
        return entities
    
    def calculate_f1_score(self, true_entities, pred_entities):
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            true_entities: List of tuples (start, end, type) for ground truth
            pred_entities: List of tuples (start, end, type) for predictions
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        # Convert to sets for easier comparison
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(true_set.intersection(pred_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_weighted_and_average_f1(self, results):
        """
        Calculate weighted F1 and average F1 across all files.
        
        Args:
            results: Dictionary with file results and overall metrics
            
        Returns:
            Dictionary with weighted_f1 and average_f1
        """
        file_results = results['file_results']
        
        if not file_results:
            return {'weighted_f1': 0, 'average_f1': 0}
        
        # Calculate average F1 (simple mean of all file F1 scores)
        total_f1 = sum(result['metrics']['f1'] for result in file_results)
        average_f1 = total_f1 / len(file_results)
        
        # Calculate weighted F1 (weighted by number of true entities in each file)
        total_true_entities = sum(len(result['true_entities']) for result in file_results)
        if total_true_entities == 0:
            weighted_f1 = 0
        else:
            weighted_f1 = sum(
                result['metrics']['f1'] * len(result['true_entities'])
                for result in file_results
            ) / total_true_entities
        
        return {
            'weighted_f1': weighted_f1,
            'average_f1': average_f1
        }
    
    def evaluate_file(self, base_name):
        """
        Evaluate a single file pair (base_name.spans and base_name.txt).
        
        Returns:
            Dictionary with evaluation metrics
        """
        spans_path = os.path.join(self.dataset_path, f"{base_name}.spans")
        txt_path = os.path.join(self.dataset_path, f"{base_name}.txt")
        
        # Check if both files exist
        if not os.path.exists(spans_path) or not os.path.exists(txt_path):
            return None
            
        # Parse ground truth entities
        true_entities = self.parse_spans_file(spans_path)
        
        # Read text and run NER
        text = self.read_text_file(txt_path)
        pred_entities = self.extract_entities_from_ner(text)
        
        # Calculate metrics
        metrics = self.calculate_f1_score(true_entities, pred_entities)
        
        return {
            'file': base_name,
            'true_entities': true_entities,
            'pred_entities': pred_entities,
            'metrics': metrics
        }
    
    def evaluate_dataset(self):
        """
        Evaluate the entire dataset.
        
        Returns:
            Dictionary with overall evaluation metrics
        """
        # Find all .spans files in the dataset
        spans_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.spans')]
        base_names = [os.path.splitext(f)[0] for f in spans_files]
        
        all_true_entities = []
        all_pred_entities = []
        file_results = []
        
        for base_name in base_names:
            result = self.evaluate_file(base_name)
            if result:
                file_results.append(result)
                all_true_entities.extend(result['true_entities'])
                all_pred_entities.extend(result['pred_entities'])
                print(f"Evaluated {base_name}: F1={result['metrics']['f1']:.4f}, "
                      f"P={result['metrics']['precision']:.4f}, "
                      f"R={result['metrics']['recall']:.4f}")
        
        # Calculate overall metrics
        overall_metrics = self.calculate_f1_score(all_true_entities, all_pred_entities)
        
        return {
            'overall_metrics': overall_metrics,
            'file_results': file_results,
            'total_files': len(file_results)
        }
    
    def print_detailed_results(self, results):
        """Print detailed evaluation results."""
        print("\n" + "="*50)
        print("NATASHA NER EVALUATION RESULTS")
        print("="*50)
        
        overall = results['overall_metrics']
        print(f"\nOverall Results ({results['total_files']} files):")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        print(f"F1 Score: {overall['f1']:.4f}")
        print(f"True Positives: {overall['true_positives']}")
        print(f"False Positives: {overall['false_positives']}")
        print(f"False Negatives: {overall['false_negatives']}")
        
        # Calculate and display weighted and average F1
        f1_metrics = self.calculate_weighted_and_average_f1(results)
        print(f"\nF1 Metrics:")
        print(f"Weighted F1: {f1_metrics['weighted_f1']:.4f}")
        print(f"Average F1: {f1_metrics['average_f1']:.4f}")
        
        # Check if the conditions are met
        weighted_condition = f1_metrics['weighted_f1'] > 0.85
        average_condition = f1_metrics['average_f1'] > 0.8
        
        print(f"\nCondition Check:")
        print(f"Weighted F1 > 0.85: {weighted_condition} ({f1_metrics['weighted_f1']:.4f})")
        print(f"Average F1 > 0.8: {average_condition} ({f1_metrics['average_f1']:.4f})")
        
        if weighted_condition and average_condition:
            print("✓ Both conditions are satisfied!")
        else:
            print("✗ One or both conditions are not satisfied.")
        
        print("\nPer-file results:")
        print("-" * 70)
        print(f"{'File':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-" * 70)
        
        for result in results['file_results']:
            file_name = result['file'][:17] + "..." if len(result['file']) > 20 else result['file']
            metrics = result['metrics']
            print(f"{file_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} {metrics['true_positives']:<5} "
                  f"{metrics['false_positives']:<5} {metrics['false_negatives']:<5}")


def main():
    # Create tester and run evaluation
    tester = FactRuEvalTester()
    
    print("Starting NatashaNER evaluation on factRuEval dataset...")
    results = tester.evaluate_dataset()
    
    # Print detailed results
    tester.print_detailed_results(results)
    
    # Calculate weighted and average F1 for final verification
    f1_metrics = tester.calculate_weighted_and_average_f1(results)
    
    # Save results to file
    import json
    with open('natasha_ner_evaluation_results.json', 'w', encoding='utf-8') as f:
        # Convert tuples to lists for JSON serialization
        serializable_results = {
            'overall_metrics': results['overall_metrics'],
            'f1_metrics': f1_metrics,
            'file_results': [
                {
                    'file': r['file'],
                    'metrics': r['metrics'],
                    'true_entities': [[e[0], e[1], e[2]] for e in r['true_entities']],
                    'pred_entities': [[e[0], e[1], e[2]] for e in r['pred_entities']]
                }
                for r in results['file_results']
            ],
            'total_files': results['total_files']
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to 'natasha_ner_evaluation_results.json'")
    
    # Return the F1 metrics for programmatic access
    return f1_metrics


if __name__ == "__main__":
    main()