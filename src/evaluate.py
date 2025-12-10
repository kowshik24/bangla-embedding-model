"""
Evaluation module for Bangla embedding model on MTEB benchmark.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
import mteb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BanglaEmbeddingEvaluator:
    """
    Evaluator for Bangla embedding models on MTEB benchmark.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./evaluation_results"
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        
        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_bengali_tasks(self) -> List[mteb.AbsTask]:
        """Get all MTEB tasks that include Bengali."""
        all_tasks = mteb.get_tasks(languages=["ben"])
        logger.info(f"Found {len(all_tasks)} Bengali tasks")
        return all_tasks
    
    def get_multilingual_tasks(self) -> List[mteb.AbsTask]:
        """Get multilingual MTEB tasks."""
        task_names = [
            "BUCC",
            "Tatoeba",
            "XNLI",
            "MultilingualSentiment",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
        ]
        
        tasks = mteb.get_tasks(tasks=task_names)
        return tasks
    
    def evaluate_on_bengali(
        self,
        task_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on Bengali MTEB tasks.
        
        Args:
            task_types: Filter by task types (e.g., ["Classification", "STS"])
        """
        logger.info("Starting Bengali MTEB evaluation...")
        
        # Get Bengali tasks
        tasks = self.get_bengali_tasks()
        
        # Filter by task type if specified
        if task_types:
            tasks = [t for t in tasks if t.metadata.type in task_types]
        
        if not tasks:
            logger.warning("No Bengali tasks found!")
            return {}
        
        # Run evaluation
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            self.model,
            output_folder=str(self.output_dir / "bengali"),
            eval_splits=["test"]
        )
        
        # Process and save results
        processed_results = self._process_results(results)
        
        results_file = self.output_dir / "bengali_results.json"
        with open(results_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        logger.info(f"Saved Bengali results to {results_file}")
        
        return processed_results
    
    def evaluate_on_multilingual(self) -> Dict:
        """Evaluate model on multilingual MTEB tasks."""
        logger.info("Starting multilingual MTEB evaluation...")
        
        tasks = self.get_multilingual_tasks()
        
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            self.model,
            output_folder=str(self.output_dir / "multilingual"),
            eval_splits=["test"]
        )
        
        processed_results = self._process_results(results)
        
        results_file = self.output_dir / "multilingual_results.json"
        with open(results_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        return processed_results
    
    def evaluate_custom_benchmark(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[float],
        benchmark_name: str = "custom"
    ) -> Dict:
        """
        Evaluate on a custom STS benchmark.
        """
        from sentence_transformers import evaluation
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=labels,
            name=benchmark_name
        )
        
        result = evaluator(self.model, output_path=str(self.output_dir))
        
        return {benchmark_name: result}
    
    def _process_results(self, results: List) -> Dict:
        """Process MTEB results into a clean format."""
        processed = {
            "model_path": self.model_path,
            "evaluation_date": datetime.now().isoformat(),
            "tasks": {},
            "summary": {}
        }
        
        scores_by_type = {}
        
        for task_result in results:
            task_name = task_result.task_name
            scores = task_result.scores
            
            processed["tasks"][task_name] = {
                "scores": scores,
                "task_type": task_result.task_type if hasattr(task_result, 'task_type') else "unknown"
            }
            
            # Aggregate by task type
            task_type = processed["tasks"][task_name]["task_type"]
            if task_type not in scores_by_type:
                scores_by_type[task_type] = []
            
            # Extract main score
            if isinstance(scores, dict) and 'test' in scores:
                main_score = scores['test'].get('main_score', 0)
                scores_by_type[task_type].append(main_score)
        
        # Calculate averages
        for task_type, type_scores in scores_by_type.items():
            if type_scores:
                processed["summary"][task_type] = {
                    "average": sum(type_scores) / len(type_scores),
                    "num_tasks": len(type_scores)
                }
        
        # Overall average
        all_scores = [s for scores in scores_by_type.values() for s in scores]
        if all_scores:
            processed["summary"]["overall"] = {
                "average": sum(all_scores) / len(all_scores),
                "num_tasks": len(all_scores)
            }
        
        return processed
    
    def generate_leaderboard_submission(self) -> Dict:
        """
        Generate files needed for MTEB leaderboard submission.
        """
        logger.info("Generating leaderboard submission files...")
        
        submission_dir = self.output_dir / "leaderboard_submission"
        submission_dir.mkdir(exist_ok=True)
        
        # Run full evaluation
        bengali_results = self.evaluate_on_bengali()
        
        # Create model card content
        model_card = f"""---
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- bangla
- bengali
- mteb
language:
- bn
- en
library_name: sentence-transformers
---

# Bangla Embedding Model

## Model Description

This is a Bangla sentence embedding model fine-tuned for semantic similarity tasks.

## MTEB Results

### Bengali Tasks

{json.dumps(bengali_results.get('summary', {}), indent=2)}

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{self.model_path}')
embeddings = model.encode(['বাংলা বাক্য'])
```

## Training

This model was trained using:
- Cross-lingual knowledge distillation
- Matryoshka representation learning
- Hard negative mining

## Citation

If you use this model, please cite:

```bibtex
@misc{{bangla-embedding-model,
  author = {{Your Name}},
  title = {{Bangla Embedding Model}},
  year = {{2024}},
  publisher = {{Hugging Face}}
}}
"""
        
        # Save model card
        with open(submission_dir / "README_MODEL_CARD.md", 'w') as f:
            f.write(model_card)
        
        # Save results in MTEB format
        with open(submission_dir / "results.json", 'w') as f:
            json.dump(bengali_results, f, indent=2)
        
        logger.info(f"Submission files saved to {submission_dir}")
        
        return {
            "submission_dir": str(submission_dir),
            "results": bengali_results
        }


def run_full_evaluation(model_path: str, output_dir: str = "./evaluation_results"):
    """Run complete MTEB evaluation for a Bangla model."""
    evaluator = BanglaEmbeddingEvaluator(model_path, output_dir)
    
    # Run Bengali evaluation
    bengali_results = evaluator.evaluate_on_bengali()
    
    # Run multilingual evaluation
    multilingual_results = evaluator.evaluate_on_multilingual()
    
    # Generate submission files
    submission = evaluator.generate_leaderboard_submission()
    
    return {
        "bengali": bengali_results,
        "multilingual": multilingual_results,
        "submission": submission
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    args = parser.parse_args()
    
    results = run_full_evaluation(args.model_path, args.output_dir)
    print(json.dumps(results, indent=2))
