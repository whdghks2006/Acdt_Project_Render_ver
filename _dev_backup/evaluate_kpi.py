# -*- coding: utf-8 -*-
"""
NER Model KPI Evaluation Script
- Evaluates trained model on test_data.json
- Calculates Precision, Recall, F1-Score per entity type
- Generates detailed KPI report
"""

import spacy
import json
import os
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================

# Test data file path
TEST_FILE = "test_data.json"

# Trained model path
MODEL_PATH = "./output/new_ner_model"

# Entity labels
LABELS = ["START_DATE", "START_TIME", "END_DATE", "END_TIME", "LOC", "EVENT_TITLE"]

# Field to label mapping (for converting JSON to entities)
FIELD_TO_LABEL = {
    "Date_Entity": "START_DATE",
    "Time_Entity": "START_TIME",
    "End_Date_Entity": "END_DATE",
    "End_Time_Entity": "END_TIME",
    "Location_Entity": "LOC",
    "Event_Entity": "EVENT_TITLE"
}


# ==============================================================================
# Data Loading Function
# ==============================================================================

def load_test_data(filepath):
    """Load and parse test data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    test_data = []
    
    for item in raw_data:
        text = item.get("Text", "")
        if not text or not text.strip():
            continue
        
        # Extract expected entities
        expected_entities = {}
        for field, label in FIELD_TO_LABEL.items():
            entity_value = item.get(field, "")
            if entity_value is not None and entity_value != "":
                entity_value = str(entity_value).strip()
                if entity_value:
                    expected_entities[label] = entity_value
        
        if expected_entities:
            test_data.append({
                "text": text,
                "expected": expected_entities
            })
    
    return test_data


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_single(nlp, text, expected):
    """
    Evaluate model prediction on a single text
    Returns: (true_positives, false_positives, false_negatives) per label
    """
    doc = nlp(text)
    
    # Get predicted entities
    predicted = {}
    for ent in doc.ents:
        if ent.label_ in LABELS:
            predicted[ent.label_] = ent.text
    
    results = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    
    for label in LABELS:
        expected_value = expected.get(label, None)
        predicted_value = predicted.get(label, None)
        
        if expected_value and predicted_value:
            # Both exist - check if they match (case-insensitive, strip whitespace)
            if expected_value.lower().strip() == predicted_value.lower().strip():
                results[label]["tp"] = 1  # True Positive
            else:
                # Predicted wrong value
                results[label]["fp"] = 1  # False Positive (wrong prediction)
                results[label]["fn"] = 1  # False Negative (missed correct value)
        elif expected_value and not predicted_value:
            results[label]["fn"] = 1  # False Negative
        elif predicted_value and not expected_value:
            results[label]["fp"] = 1  # False Positive
    
    return results


def calculate_metrics(total_results):
    """Calculate Precision, Recall, F1 from aggregated results"""
    metrics = {}
    
    for label in LABELS:
        tp = total_results[label]["tp"]
        fp = total_results[label]["fp"]
        fn = total_results[label]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }
    
    # Calculate overall metrics (micro-average)
    total_tp = sum(total_results[label]["tp"] for label in LABELS)
    total_fp = sum(total_results[label]["fp"] for label in LABELS)
    total_fn = sum(total_results[label]["fn"] for label in LABELS)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics["OVERALL"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall_precision * 100,
        "recall": overall_recall * 100,
        "f1": overall_f1 * 100
    }
    
    return metrics


# ==============================================================================
# Report Generation
# ==============================================================================

def print_report(metrics, test_count):
    """Print KPI report"""
    print("\n" + "=" * 70)
    print("                     NER MODEL KPI EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nTest samples: {test_count}")
    print(f"Entity labels: {len(LABELS)}")
    
    print("\n" + "-" * 70)
    print(f"{'Entity':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<15} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
              f"{m['precision']:>11.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%")
    
    print("-" * 70)
    m = metrics["OVERALL"]
    print(f"{'OVERALL':<15} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
          f"{m['precision']:>11.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%")
    print("=" * 70)
    
    # KPI Summary
    print("\n" + "=" * 70)
    print("                           KPI SUMMARY")
    print("=" * 70)
    print(f"\n  Overall Precision: {m['precision']:.2f}%")
    print(f"  Overall Recall:    {m['recall']:.2f}%")
    print(f"  Overall F1-Score:  {m['f1']:.2f}%")
    print("\n" + "=" * 70)


def save_report(metrics, test_count, output_path="kpi_report.txt"):
    """Save KPI report to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("                     NER MODEL KPI EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Test samples: {test_count}\n")
        f.write(f"Entity labels: {len(LABELS)}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Entity':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 70 + "\n")
        
        for label in LABELS:
            m = metrics[label]
            f.write(f"{label:<15} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
                    f"{m['precision']:>11.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%\n")
        
        f.write("-" * 70 + "\n")
        m = metrics["OVERALL"]
        f.write(f"{'OVERALL':<15} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
                f"{m['precision']:>11.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%\n")
        f.write("=" * 70 + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("                           KPI SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Overall Precision: {m['precision']:.2f}%\n")
        f.write(f"  Overall Recall:    {m['recall']:.2f}%\n")
        f.write(f"  Overall F1-Score:  {m['f1']:.2f}%\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n[INFO] Report saved to: {output_path}")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NER Model KPI Evaluation")
    print("=" * 60)
    
    # Check files
    if not os.path.exists(TEST_FILE):
        print(f"[ERROR] Test file not found: {TEST_FILE}")
        exit()
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[TIP] Run train_improved.py first to train the model")
        exit()
    
    # Load model
    print(f"\n[Step 1] Loading model from {MODEL_PATH}...")
    nlp = spacy.load(MODEL_PATH)
    print("[INFO] Model loaded successfully")
    
    # Load test data
    print(f"\n[Step 2] Loading test data from {TEST_FILE}...")
    test_data = load_test_data(TEST_FILE)
    print(f"[INFO] Loaded {len(test_data)} test samples")
    
    # Evaluate
    print("\n[Step 3] Evaluating...")
    total_results = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    
    for i, item in enumerate(test_data):
        results = evaluate_single(nlp, item["text"], item["expected"])
        
        for label in LABELS:
            total_results[label]["tp"] += results[label]["tp"]
            total_results[label]["fp"] += results[label]["fp"]
            total_results[label]["fn"] += results[label]["fn"]
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1}/{len(test_data)}...")
    
    # Calculate metrics
    metrics = calculate_metrics(total_results)
    
    # Print report
    print_report(metrics, len(test_data))
    
    # Save report
    save_report(metrics, len(test_data))
    
    print("\nEvaluation Complete!")
