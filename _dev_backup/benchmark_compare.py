# -*- coding: utf-8 -*-
"""
NER Model Comparison Benchmark
- Compares Custom NER model (v1, v2) vs Baseline (en_core_web_md)
- Uses test_data.json for fair comparison
"""

import spacy
import json
import os
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

TEST_FILE = "test_data.json"
CUSTOM_MODEL_V1_PATH = "../output/new_ner_model"      # Current production
CUSTOM_MODEL_V2_PATH = "../output/new_ner_model_v2"   # Newly trained with augmented data
BASELINE_MODEL = "en_core_web_md"

# Field to label mapping
FIELD_TO_LABEL = {
    "Date_Entity": "START_DATE",
    "Time_Entity": "START_TIME",
    "End_Date_Entity": "END_DATE",
    "End_Time_Entity": "END_TIME",
    "Location_Entity": "LOC",
    "Event_Entity": "EVENT_TITLE"
}

# For baseline model (uses different labels)
BASELINE_LABEL_MAP = {
    "DATE": ["START_DATE", "END_DATE"],
    "TIME": ["START_TIME", "END_TIME"],
    "LOC": ["LOC"],
    "GPE": ["LOC"],
    "EVENT": ["EVENT_TITLE"]
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_test_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    test_data = []
    for item in raw_data:
        text = item.get("Text", "")
        if not text or not text.strip():
            continue
        
        expected = {}
        for field, label in FIELD_TO_LABEL.items():
            value = item.get(field, "")
            if value is not None and value != "":
                value = str(value).strip()
                if value:
                    expected[label] = value
        
        if expected:
            test_data.append({"text": text, "expected": expected})
    
    return test_data


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_custom_model(nlp, test_data):
    """Evaluate custom NER model (6 labels)"""
    labels = ["START_DATE", "START_TIME", "END_DATE", "END_TIME", "LOC", "EVENT_TITLE"]
    results = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    
    for item in test_data:
        doc = nlp(item["text"])
        predicted = {}
        for ent in doc.ents:
            if ent.label_ in labels:
                predicted[ent.label_] = ent.text
        
        for label in labels:
            expected_val = item["expected"].get(label)
            predicted_val = predicted.get(label)
            
            if expected_val and predicted_val:
                # Allow partial matching
                if expected_val.lower().strip() in predicted_val.lower().strip() or \
                   predicted_val.lower().strip() in expected_val.lower().strip():
                    results[label]["tp"] += 1
                else:
                    results[label]["fp"] += 1
                    results[label]["fn"] += 1
            elif expected_val and not predicted_val:
                results[label]["fn"] += 1
            elif predicted_val and not expected_val:
                results[label]["fp"] += 1
    
    return results


def evaluate_baseline_model(nlp, test_data):
    """Evaluate baseline model (DATE, TIME, LOC, etc.)"""
    labels = ["START_DATE", "START_TIME", "END_DATE", "END_TIME", "LOC", "EVENT_TITLE"]
    results = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    
    for item in test_data:
        doc = nlp(item["text"])
        
        # Extract entities from baseline model
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
        locs = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
        events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]
        
        # Map to our labels (baseline can only detect DATE/TIME, not START/END)
        predicted = {}
        if dates:
            predicted["START_DATE"] = dates[0]
        if times:
            predicted["START_TIME"] = times[0]
        if locs:
            predicted["LOC"] = locs[0]
        if events:
            predicted["EVENT_TITLE"] = events[0]
        
        for label in labels:
            expected_val = item["expected"].get(label)
            predicted_val = predicted.get(label)
            
            if expected_val and predicted_val:
                if expected_val.lower().strip() in predicted_val.lower().strip() or \
                   predicted_val.lower().strip() in expected_val.lower().strip():
                    results[label]["tp"] += 1
                else:
                    results[label]["fp"] += 1
                    results[label]["fn"] += 1
            elif expected_val and not predicted_val:
                results[label]["fn"] += 1
            elif predicted_val and not expected_val:
                results[label]["fp"] += 1
    
    return results


def calculate_metrics(results):
    """Calculate P/R/F1 from results"""
    labels = list(results.keys())
    metrics = {}
    
    for label in labels:
        tp = results[label]["tp"]
        fp = results[label]["fp"]
        fn = results[label]["fn"]
        
        p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        metrics[label] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    
    # Overall
    total_tp = sum(results[l]["tp"] for l in labels)
    total_fp = sum(results[l]["fp"] for l in labels)
    total_fn = sum(results[l]["fn"] for l in labels)
    
    p = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    metrics["OVERALL"] = {"precision": p, "recall": r, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}
    
    return metrics


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("              NER MODEL COMPARISON BENCHMARK")
    print("=" * 70)
    
    # Load test data
    print("\n[1] Loading test data...")
    test_data = load_test_data(TEST_FILE)
    print(f"    Loaded {len(test_data)} test samples")
    
    # Load models
    print("\n[2] Loading models...")
    
    all_metrics = {}
    
    # Baseline
    print(f"    Loading Baseline Model: {BASELINE_MODEL}")
    baseline_nlp = spacy.load(BASELINE_MODEL)
    print("\n[3] Evaluating Baseline Model...")
    baseline_results = evaluate_baseline_model(baseline_nlp, test_data)
    all_metrics["Baseline"] = calculate_metrics(baseline_results)
    
    # Custom v1
    if os.path.exists(CUSTOM_MODEL_V1_PATH):
        print(f"    Loading Custom Model v1: {CUSTOM_MODEL_V1_PATH}")
        v1_nlp = spacy.load(CUSTOM_MODEL_V1_PATH)
        print("\n[4] Evaluating Custom Model v1...")
        v1_results = evaluate_custom_model(v1_nlp, test_data)
        all_metrics["Custom v1"] = calculate_metrics(v1_results)
    else:
        print(f"    [SKIP] Custom v1 not found: {CUSTOM_MODEL_V1_PATH}")
    
    # Custom v2
    if os.path.exists(CUSTOM_MODEL_V2_PATH):
        print(f"    Loading Custom Model v2: {CUSTOM_MODEL_V2_PATH}")
        v2_nlp = spacy.load(CUSTOM_MODEL_V2_PATH)
        print("\n[5] Evaluating Custom Model v2...")
        v2_results = evaluate_custom_model(v2_nlp, test_data)
        all_metrics["Custom v2"] = calculate_metrics(v2_results)
    else:
        print(f"    [SKIP] Custom v2 not found: {CUSTOM_MODEL_V2_PATH}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("                        COMPARISON RESULTS")
    print("=" * 70)
    
    labels = ["START_DATE", "START_TIME", "END_DATE", "END_TIME", "LOC", "EVENT_TITLE", "OVERALL"]
    model_names = list(all_metrics.keys())
    
    # Header
    header = f"{'Entity':<15}"
    for name in model_names:
        header += f" | {name + ' F1':>12}"
    if "Custom v1" in model_names and "Custom v2" in model_names:
        header += f" | {'v1→v2':>10}"
    print(f"\n{header}")
    print("-" * 70)
    
    # Rows
    for label in labels:
        row = f"{label:<15}"
        for name in model_names:
            f1 = all_metrics[name][label]["f1"]
            row += f" | {f1:>11.1f}%"
        
        # v1 → v2 diff
        if "Custom v1" in model_names and "Custom v2" in model_names:
            diff = all_metrics["Custom v2"][label]["f1"] - all_metrics["Custom v1"][label]["f1"]
            marker = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➖")
            row += f" | {diff:>+8.1f}% {marker}"
        
        print(row)
    
    print("=" * 70)
    
    # Summary
    print(f"\n📊 SUMMARY")
    for name in model_names:
        print(f"   {name} Overall F1: {all_metrics[name]['OVERALL']['f1']:.1f}%")
    
    if "Custom v1" in all_metrics and "Custom v2" in all_metrics:
        v1_f1 = all_metrics["Custom v1"]["OVERALL"]["f1"]
        v2_f1 = all_metrics["Custom v2"]["OVERALL"]["f1"]
        diff = v2_f1 - v1_f1
        
        print(f"\n   v1 → v2 Improvement: {diff:+.1f}%")
        
        if diff > 1:
            print("\n🎉 Custom v2 shows improvement! Consider replacing v1.")
        elif diff > 0:
            print("\n✅ Custom v2 shows slight improvement.")
        elif diff == 0:
            print("\n➖ No difference between v1 and v2.")
        else:
            print("\n⚠️ Custom v1 performs better. Keep current model.")
    
    print("\n" + "=" * 70)
    print("                           MODEL INFO")
    print("=" * 70)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Test Samples: {len(test_data)}")
    print(f"\nBaseline: {BASELINE_MODEL}")
    print(f"Custom v1: {CUSTOM_MODEL_V1_PATH}")
    print(f"Custom v2: {CUSTOM_MODEL_V2_PATH} (augmented data: 15,508 samples)")
    print("\n" + "=" * 70)
