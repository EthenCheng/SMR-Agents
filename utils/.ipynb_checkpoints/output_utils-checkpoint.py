"""Output utility functions"""
import json
import os
from typing import Any, List

def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def format_output_filepath(language_model: str, visual_model: str, 
                          method: str, dataset: str) -> str:
    """Format output file path"""
    filename = f"{dataset}_{method}_{language_model}_{visual_model}_results.json"
    return os.path.join("outputs", filename)

def format_json_out_put(question: str, ground_truth: str, prediction: str, 
                       idx: int, output_path: str):
    """Format and save JSON output"""
    result = {
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction
    }
    
    # Load existing results if file exists
    results = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
    
    # Update or append result
    found = False
    for i, r in enumerate(results):
        if r["index"] == idx:
            results[i] = result
            found = True
            break
    
    if not found:
        results.append(result)
    
    # Sort by index
    results.sort(key=lambda x: x["index"])
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def filter_finished(total_count: int, output_path: str) -> List[int]:
    """Filter out already processed indices"""
    if not os.path.exists(output_path):
        return list(range(total_count))
    
    with open(output_path, 'r') as f:
        results = json.load(f)
    
    finished_indices = {r["index"] for r in results}
    return [i for i in range(total_count) if i not in finished_indices]
