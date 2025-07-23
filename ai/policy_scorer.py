# ai/policy_scorer.py
"""
NLP Policy Impact Scorer - Production Version
Implements a zero-shot classification approach to analyze legislative text and
estimate its thematic framing across the five agency domains. This version includes
robust text segmentation and configurable analysis parameters.
"""
# --- Library Imports ---
import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional
from transformers import pipeline
import torch
from datetime import datetime
import json

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Initializer for Hugging Face Pipeline ---
def initialize_classifier():
    """Initializes the zero-shot classifier, handling device placement."""
    try:
        # Use GPU if available for significantly faster processing
        device = 0 if torch.cuda.is_available() else -1
        model_name = "facebook/bart-large-mnli"
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        logger.info(f"Zero-shot classifier '{model_name}' loaded successfully on {'GPU' if device == 0 else 'CPU'}.")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load Hugging Face classifier: {e}")
        return None

# Initialize classifier once on module load to avoid reloading the model on each call
CLASSIFIER = initialize_classifier()

CANDIDATE_LABELS = {
    "Economic": {
        "positive": "expands economic opportunity and freedom",
        "negative": "reduces economic security and resources"
    },
    "Health": {
        "positive": "improves public health and healthcare access",
        "negative": "harms public health and safety"
    },
    "Political": {
        "positive": "increases political freedom and participation",
        "negative": "decreases political rights and accountability"
    },
    "Social": {
        "positive": "strengthens social trust and cohesion",
        "negative": "weakens social bonds and increases division"
    },
    "Educational": {
        "positive": "enhances educational access and quality",
        "negative": "damages educational opportunity"
    }
}

class PolicyScorer:
    """A zero-shot policy impact scorer for analyzing legislative text."""
    def __init__(self):
        if CLASSIFIER is None:
            raise RuntimeError("Classifier could not be initialized. Cannot create PolicyScorer.")
        self.classifier = CLASSIFIER
        self.candidate_labels = CANDIDATE_LABELS
        self.all_labels = [label for domain in self.candidate_labels.values() for label in domain.values()]
        # FIX: Initialize scoring_history as an empty list
        self.scoring_history = []

    def _segment_text(self, text: str, max_chunk_length: int) -> List[str]:
        """Segments a long document into meaningful chunks within token limits."""
        text = re.sub(r'\s+', ' ', text).strip()
        # Split by paragraphs as a robust baseline
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Further split chunks that are too long
        # FIX: Initialize final_chunks as an empty list
        final_chunks = []
        for chunk in chunks:
            # Check if chunk exceeds the tokenizer's max length
            if len(self.classifier.tokenizer.tokenize(chunk)) > max_chunk_length:
                # Simple split by sentences for oversized chunks
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_chunk = ""
                for sentence in sentences:
                    if len(self.classifier.tokenizer.tokenize(current_chunk + sentence)) <= max_chunk_length:
                        current_chunk += sentence + " "
                    else:
                        final_chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Segmented text into {len(final_chunks)} chunks for analysis.")
        return final_chunks

    def score_policy_text(self, text: str, policy_name: Optional[str] = "Untitled Policy",
                          confidence_threshold: float = 0.5, normalization: str = 'density',
                          max_chunk_length: int = 512, min_chunk_length: int = 25) -> Dict:
        """Analyzes a policy document and provides a net impact score for each domain."""
        start_time = datetime.now()
        # Filter out chunks that are too short to be meaningful
        chunks = [c for c in self._segment_text(text, max_chunk_length) if len(c.split()) > min_chunk_length / 5]
        if not chunks:
            return {'error': 'No valid text chunks found for analysis.'}

        domain_scores = defaultdict(lambda: defaultdict(float))
        impactful_chunks_count = 0

        for chunk in chunks:
            try:
                # The model predicts the most likely label for the chunk
                result = self.classifier(chunk, self.all_labels, multi_label=False)
                top_label = result['labels'][0] # In single-label mode, access the first element
                top_score = result['scores'][0]

                if top_score >= confidence_threshold:
                    impactful_chunks_count += 1
                    # Find which domain the top label belongs to and add its score
                    for domain, labels in self.candidate_labels.items():
                        if top_label == labels["positive"]:
                            domain_scores[domain]["positive"] += top_score
                        elif top_label == labels["negative"]:
                            domain_scores[domain]["negative"] += top_score
            except Exception as e:
                logger.warning(f"Skipping chunk due to classification error: {e}")

        # --- Calculate Net Impact Scores ---
        net_impact_scores = {}
        # Determine the normalization factor based on the chosen strategy
        if normalization == 'magnitude' and impactful_chunks_count > 0:
            normalizer = impactful_chunks_count
        else: # Default to 'density' normalization
            normalizer = len(chunks) if chunks else 1

        for domain in self.candidate_labels.keys():
            scores = domain_scores[domain]
            total_positive = scores["positive"] / normalizer
            total_negative = scores["negative"] / normalizer
            net_impact_scores[domain] = total_positive - total_negative

        analysis_record = {
            'policy_name': policy_name,
            'timestamp': datetime.now().isoformat(),
            'net_scores': net_impact_scores,
            'processing_time_sec': (datetime.now() - start_time).total_seconds(),
            'config': {'confidence_threshold': confidence_threshold, 'normalization': normalization}
        }
        self.scoring_history.append(analysis_record)
        
        logger.info(f"Analysis for '{policy_name}' complete. Net scores: {net_impact_scores}")
        return analysis_record

# --- Convenience function for API integration ---
def score_policy_text(text: str, **kwargs) -> Dict:
    """
    Convenience function to create a PolicyScorer instance and score text.
    This is the primary entry point for the API.
    """
    try:
        scorer = PolicyScorer()
        return scorer.score_policy_text(text, **kwargs)
    except RuntimeError as e:
        # This catches the case where the global CLASSIFIER failed to initialize
        return {'error': str(e)}