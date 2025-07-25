import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyScorer:
    """Zero-shot policy impact scorer for legislative text analysis."""
    CANDIDATE_LABELS = {
        "Economic": {
            "positive": "expands economic opportunity and freedom",
            "negative": "reduces economic security and resources"
        },
        "Political": {
            "positive": "increases political freedom and participation",
            "negative": "decreases political rights and accountability"
        },
        "Social": {
            "positive": "strengthens social trust and cohesion",
            "negative": "weakens social bonds and increases division"
        },
        "Health": {
            "positive": "improves public health and healthcare access",
            "negative": "harms public health and safety"
        },
        "Educational": {
            "positive": "enhances educational access and quality",
            "negative": "damages educational opportunity"
        }
    }

    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = -1, batch_size: int = 8, max_workers: int = 4):
        self.model_name = model_name
        self.classifier = pipeline('zero-shot-classification', model=model_name, device=device)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.all_labels = [lbl for sub in self.CANDIDATE_LABELS.values() for lbl in sub.values()]
        self.trainer = None  # For fine-tuning

    def _segment_text(self, text: str, max_chunk_length: int = 512, min_chunk_length: int = 25, overlap: int = 128) -> List[str]:
        """Segment text into overlapping chunks."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be non-empty string.")
        text = re.sub(r'\n\s*\n', '||', text)
        chunks = [chunk.strip() for chunk in text.split('||') if len(chunk.strip()) >= min_chunk_length]
        overlapped_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_length:
                for start in range(0, len(chunk), max_chunk_length - overlap):
                    end = start + max_chunk_length
                    overlapped_chunks.append(chunk[start:end])
            else:
                overlapped_chunks.append(chunk)
        return overlapped_chunks

    def score_policy_text(self, text: str, confidence_threshold: float = 0.5, normalization: str = 'density', max_chunk_length: int = 512, min_chunk_length: int = 25) -> Dict[str, float]:
        """Analyze policy text and compute net impact scores per domain."""
        start_time = datetime.now()
        logger.info("Segmenting policy text...")
        chunks = self._segment_text(text, max_chunk_length, min_chunk_length)
        if not chunks:
            logger.warning("No valid chunks found.")
            return {domain: 0.0 for domain in self.CANDIDATE_LABELS}

        logger.info(f"Analyzing {len(chunks)} chunks asynchronously...")
        domain_scores = defaultdict(lambda: defaultdict(float))

        def classify_batch(batch: List[str]) -> List[Dict]:
            try:
                return self.classifier(batch, self.all_labels, multi_label=True)
            except Exception as e:
                logger.error(f"Classification error: {e}")
                return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(classify_batch, chunks[i:i + self.batch_size]) for i in range(0, len(chunks), self.batch_size)]
            for future in as_completed(futures):
                results = future.result()
                for res in results:
                    if isinstance(res, dict):
                        res = [res]  # Handle single result
                    for r in res:
                        labels, scores = r['labels'], r['scores']
                        for label, score in zip(labels, scores):
                            if score < confidence_threshold:
                                continue
                            for domain, domain_labels in self.CANDIDATE_LABELS.items():
                                if label == domain_labels['positive']:
                                    domain_scores[domain]['positive'] += score
                                elif label == domain_labels['negative']:
                                    domain_scores[domain]['negative'] += score

        logger.info("Aggregating results...")
        net_impact_scores = {}
        normalizer = len(chunks) if normalization == 'density' else 1.0
        for domain, scores in domain_scores.items():
            total_positive = scores['positive'] / normalizer
            total_negative = scores['negative'] / normalizer
            net_impact_scores[domain] = total_positive - total_negative

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing complete in {processing_time:.2f} seconds.")
        return net_impact_scores

    def export_for_iqa(self, text: str, scores: Dict[str, float], output_file: str = 'iqa_export.json') -> None:
        """Export scores and text for Integrated Qualitative Analysis (IQA)."""
        export_data = {
            'text': text,
            'net_impact_scores': scores,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=4)
        logger.info(f"Exported for IQA to {output_file}")

    def fine_tune(self, train_data: List[Dict], output_dir: str = './fine_tuned_model'):
        """Fine-tune model if labeled data available (placeholder using Trainer)."""
        if not train_data:
            raise ValueError("No training data provided.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.all_labels))
        # Prepare dataset (assume train_data is [{'text': str, 'labels': [int]}])
        # ... (Implement dataset prep; skipped for brevity)
        training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=3)
        self.trainer = Trainer(model=model, args=training_args)  # Add dataset
        self.trainer.train()
        self.classifier = pipeline('zero-shot-classification', model=output_dir)

# Example usage
if __name__ == '__main__':
    scorer = PolicyScorer(model_name='microsoft/deberta-v3-base')  # Experiment with DeBERTa
    sample_text = "Policy text here..."  # Replace
    scores = scorer.score_policy_text(sample_text)
    scorer.export_for_iqa(sample_text, scores)
    print(scores)