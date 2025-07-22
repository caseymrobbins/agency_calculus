# ai/policy_scorer.py
"""
NLP Policy Impact Scorer
Implements Task 3.3 of the Agency Monitor project

Zero-shot classification approach to analyze legislative text and
estimate its potential impact on the five agency domains.
"""

# --- Library Imports ---
import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from transformers import pipeline
import torch
from datetime import datetime
import json

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    logger.info("GPU detected. Using GPU for inference.")
else:
    logger.info("No GPU available. Using CPU for inference.")

# Initialize the zero-shot classification pipeline
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    logger.info("Zero-shot classifier loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load classifier: {e}")
    raise

# Table 2: Candidate Labels for NLP Policy Scorer
# These labels are crafted as opposing pairs to facilitate a net impact score.
# They are derived from the definitions of agency in the AC4.3 framework.
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

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence to consider a classification
MAX_CHUNK_LENGTH = 512  # Maximum tokens per chunk (BART limitation)
MIN_CHUNK_LENGTH = 50   # Minimum meaningful chunk size


class PolicyScorer:
    """
    A zero-shot policy impact scorer for analyzing legislative text.
    """
    
    def __init__(self, 
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 candidate_labels: Dict = None):
        """
        Initialize the PolicyScorer.
        
        Args:
            confidence_threshold: Minimum confidence score to consider
            candidate_labels: Custom labels dict (uses default if None)
        """
        self.confidence_threshold = confidence_threshold
        self.candidate_labels = candidate_labels or CANDIDATE_LABELS
        
        # Flatten labels for classifier
        self.all_labels = []
        for domain in self.candidate_labels.values():
            self.all_labels.extend(domain.values())
        
        # Track scoring history for analysis
        self.scoring_history = []
        
    def _segment_text(self, text: str) -> List[str]:
        """
        Segments a long document into meaningful chunks.
        
        Enhanced segmentation that:
        1. Respects paragraph boundaries
        2. Handles various document formats
        3. Ensures chunks are within token limits
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        # Normalize whitespace and line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split on section markers commonly found in legislation
        section_patterns = [
            r'\n(?=Section \d+)',  # Section headers
            r'\n(?=SECTION \d+)',
            r'\n(?=Article \d+)',
            r'\n(?=ARTICLE \d+)',
            r'\n(?=\d+\.\s)',      # Numbered sections
            r'\n\n',               # Double newlines (paragraphs)
        ]
        
        chunks = [text]
        for pattern in section_patterns:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) > MAX_CHUNK_LENGTH:
                    splits = re.split(pattern, chunk)
                    new_chunks.extend(splits)
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
        
        # Clean and filter chunks
        cleaned_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                # If still too long, truncate intelligently at sentence boundary
                if len(chunk) > MAX_CHUNK_LENGTH:
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= MAX_CHUNK_LENGTH:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                cleaned_chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    if current_chunk:
                        cleaned_chunks.append(current_chunk.strip())
                else:
                    cleaned_chunks.append(chunk)
        
        logger.info(f"Segmented text into {len(cleaned_chunks)} chunks")
        return cleaned_chunks

    def _classify_chunk(self, chunk: str, confidence_threshold: float = None) -> Dict[str, float]:
        """
        Classify a single chunk of text.
        
        Args:
            chunk: Text chunk to classify
            confidence_threshold: Minimum confidence (uses instance default if None)
            
        Returns:
            Dictionary of label scores
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        try:
            result = classifier(
                chunk, 
                self.all_labels, 
                multi_label=False,
                truncation=True
            )
            
            # Convert to dict for easier processing
            scores = {}
            for label, score in zip(result['labels'], result['scores']):
                if score >= confidence_threshold:
                    scores[label] = score
                    
            return scores
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {}

    def score_policy_text(self, 
                         text: str, 
                         policy_name: Optional[str] = None,
                         confidence_threshold: float = CONFIDENCE_THRESHOLD,
                         normalization: str = 'density') -> Dict[str, float]:
        """
        Analyzes a policy document and provides a net impact score for each AC4 domain.

        Args:
            text (str): The full text of the policy or legislative document.
            policy_name (str, optional): Name/identifier for the policy
            confidence_threshold (float): Minimum confidence score to consider a classification
            normalization (str): Method for normalization. 'density' (default) normalizes 
                               by all chunks. 'magnitude' normalizes by chunks with impact.

        Returns:
            dict: A dictionary containing the net impact score for each of the five domains.
                  Scores range from -1.0 (very negative) to +1.0 (very positive)
        """
        start_time = datetime.now()
        logger.info(f"Starting policy analysis{f' for {policy_name}' if policy_name else ''}")
        
        # Segment text
        chunks = self._segment_text(text)
        if not chunks:
            logger.warning("No valid text chunks found")
            return {domain: 0.0 for domain in self.candidate_labels.keys()}
        
        # Track scores by domain
        domain_scores = defaultdict(lambda: defaultdict(float))
        chunk_classifications = []
        
        # Classify each chunk
        logger.info(f"Running zero-shot classification on chunks (confidence threshold: {confidence_threshold})...")
        for i, chunk in enumerate(chunks):
            chunk_scores = self._classify_chunk(chunk, confidence_threshold)
            
            if chunk_scores:
                chunk_classifications.append({
                    'chunk_index': i,
                    'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                    'scores': chunk_scores
                })
                
                # Aggregate scores by domain
                for label, score in chunk_scores.items():
                    for domain_name, domain_labels in self.candidate_labels.items():
                        if label == domain_labels.get("positive"):
                            domain_scores[domain_name]["positive"] += score
                        elif label == domain_labels.get("negative"):
                            domain_scores[domain_name]["negative"] += score
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        # Calculate net impact scores
        logger.info(f"Calculating net impact scores using '{normalization}' normalization...")
        net_impact_scores = {}
        score_details = {}
        
        # Count chunks with impact for magnitude normalization
        impactful_chunk_count = len(chunk_classifications)
        
        for domain, scores in domain_scores.items():
            # Choose normalization method
            if normalization == 'magnitude' and impactful_chunk_count > 0:
                normalizer = impactful_chunk_count
            else:  # Default to 'density' or if no chunks were impactful
                normalizer = len(chunks) if chunks else 1
            
            # Normalize scores
            total_positive = scores["positive"] / normalizer
            total_negative = scores["negative"] / normalizer
            
            # Net score: difference between positive and negative
            net_score = total_positive - total_negative
            net_impact_scores[domain] = round(net_score, 4)
            
            # Store detailed scoring for transparency
            score_details[domain] = {
                'positive_score': round(total_positive, 4),
                'negative_score': round(total_negative, 4),
                'net_score': round(net_score, 4),
                'chunks_with_impact': sum(1 for c in chunk_classifications 
                                        if any(self.candidate_labels[domain][pol] in c['scores'] 
                                              for pol in ['positive', 'negative'])),
                'normalization_method': normalization,
                'normalizer': normalizer
            }
        
        # Create analysis record
        analysis_record = {
            'timestamp': datetime.now().isoformat(),
            'policy_name': policy_name,
            'text_length': len(text),
            'chunk_count': len(chunks),
            'impactful_chunks': impactful_chunk_count,
            'confidence_threshold': confidence_threshold,
            'normalization': normalization,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'net_scores': net_impact_scores,
            'score_details': score_details,
            'top_chunks': chunk_classifications[:5]  # Store top 5 most confident chunks
        }
        
        self.scoring_history.append(analysis_record)
        
        # Log results
        logger.info(f"Analysis complete in {analysis_record['processing_time']:.2f} seconds")
        for domain, score in net_impact_scores.items():
            direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            logger.info(f"{domain}: {score:+.4f} ({direction})")
        
        return net_impact_scores

    def get_detailed_analysis(self, text: str, policy_name: Optional[str] = None) -> Dict:
        """
        Provides detailed analysis including chunk-by-chunk scoring.
        
        Args:
            text: Policy text to analyze
            policy_name: Optional policy identifier
            
        Returns:
            Detailed analysis dictionary
        """
        # First get the net scores
        net_scores = self.score_policy_text(text, policy_name)
        
        # Return the most recent detailed analysis
        if self.scoring_history:
            return self.scoring_history[-1]
        
        return {'net_scores': net_scores}

    def export_analysis(self, filepath: str):
        """
        Export scoring history to JSON file.
        
        Args:
            filepath: Path to save the analysis
        """
        with open(filepath, 'w') as f:
            json.dump(self.scoring_history, f, indent=2)
        logger.info(f"Analysis exported to {filepath}")


# --- Convenience function for backwards compatibility ---
def score_policy_text(text: str, 
                     confidence_threshold: float = CONFIDENCE_THRESHOLD,
                     normalization: str = 'density') -> Dict[str, float]:
    """
    Analyzes a policy document and provides a net impact score for each AC4 domain.
    
    This is a convenience function that creates a PolicyScorer instance
    and scores the text.

    Args:
        text (str): The full text of the policy or legislative document.
        confidence_threshold (float): Minimum confidence score to consider
        normalization (str): 'density' or 'magnitude' normalization

    Returns:
        dict: A dictionary containing the net impact score for each of the five domains.
    """
    scorer = PolicyScorer()
    return scorer.score_policy_text(text, confidence_threshold=confidence_threshold, normalization=normalization)


# --- Example Usage ---
def main():
    """
    Demonstrates the PolicyScorer with various policy examples.
    """
    logger.info("=== Policy Impact Scorer Demo ===")
    
    # Initialize scorer
    scorer = PolicyScorer()
    
    # Test Case 1: The "One Big Beautiful Bill Act" (from project requirements)
    logger.info("\nTest Case 1: One Big Beautiful Bill Act")
    one_big_bill_text = """
    A Bill to promote economic growth and ensure national security.
    
    Section 1: Economic Stimulus through Tariffs
    To stimulate the economy, this act authorizes a 25% tariff on all imported goods. 
    This measure is designed to protect domestic industries, encourage local manufacturing, 
    and create jobs for our citizens. It will ensure a level playing field and reduce 
    our reliance on foreign supply chains.
    
    Section 2: Healthcare Innovation
    In the interest of public health, regulations on experimental medical treatments 
    will be streamlined to accelerate innovation. This will provide faster access to 
    potentially life-saving therapies for patients in need.
    
    Section 3: National Unity through Service
    To foster social cohesion, this act establishes a National Service Program, 
    requiring all citizens aged 18-25 to complete one year of service in either 
    military or community development roles. This will build a shared sense of 
    identity and purpose.
    
    Section 4: Educational Standards
    To ensure educational excellence, national standardized testing will be 
    mandatory for all students. Schools that fail to meet benchmarks will face 
    reduced funding until improvements are demonstrated.
    """
    
    scores1 = scorer.score_policy_text(one_big_bill_text, "One Big Beautiful Bill Act")
    
    # Test Case 2: Progressive Policy Example
    logger.info("\nTest Case 2: Universal Opportunity Act")
    progressive_text = """
    The Universal Opportunity Act
    
    Title I: Economic Justice
    This act establishes a universal basic income program to ensure economic 
    security for all citizens. Additionally, it creates a federal jobs guarantee 
    program focused on green infrastructure development.
    
    Title II: Healthcare as a Right
    All residents shall have access to comprehensive healthcare services through 
    a single-payer Medicare for All system. Prescription drug prices will be 
    negotiated to ensure affordability.
    
    Title III: Democratic Participation
    Automatic voter registration and mail-in ballot access will be guaranteed 
    to all eligible citizens. Campaign finance will be publicly funded to reduce 
    the influence of special interests.
    
    Title IV: Educational Opportunity
    Free public college and trade school education will be available to all. 
    Student debt forgiveness programs will relieve the burden on current borrowers.
    """
    
    scores2 = scorer.score_policy_text(progressive_text, "Universal Opportunity Act")
    
    # Test Case 3: Neutral/Technical Policy
    logger.info("\nTest Case 3: Infrastructure Modernization Act")
    technical_text = """
    Infrastructure Modernization Act of 2025
    
    Section 1: Definitions
    For the purposes of this act, "infrastructure" shall include roads, bridges, 
    airports, seaports, rail systems, telecommunications networks, and utility systems.
    
    Section 2: Funding Allocation
    $500 billion shall be allocated over 10 years for infrastructure improvements, 
    distributed according to population and need-based formulas established by 
    the Department of Transportation.
    
    Section 3: Project Approval Process
    All projects exceeding $100 million must undergo environmental impact assessment 
    and community consultation processes as defined in existing regulations.
    
    Section 4: Reporting Requirements
    Quarterly reports on project progress and budget utilization shall be submitted 
    to Congress by implementing agencies.
    """
    
    scores3 = scorer.score_policy_text(technical_text, "Infrastructure Modernization Act")
    
    # Comparative Analysis
    logger.info("\n=== Comparative Analysis ===")
    
    policies = [
        ("One Big Beautiful Bill Act", scores1),
        ("Universal Opportunity Act", scores2),
        ("Infrastructure Modernization Act", scores3)
    ]
    
    # Create comparison table
    domains = list(CANDIDATE_LABELS.keys())
    
    print("\nPolicy Impact Comparison:")
    print("-" * 80)
    print(f"{'Policy':<35} | " + " | ".join(f"{d[:4]:^7}" for d in domains))
    print("-" * 80)
    
    for policy_name, scores in policies:
        score_strs = []
        for domain in domains:
            score = scores[domain]
            # Color coding would be nice but keeping it simple
            if score > 0.1:
                score_str = f"+{score:.3f}"
            elif score < -0.1:
                score_str = f"{score:.3f}"
            else:
                score_str = f" {score:.3f}"
            score_strs.append(score_str)
        
        print(f"{policy_name:<35} | " + " | ".join(f"{s:^7}" for s in score_strs))
    
    # Export analysis
    logger.info("\nExporting detailed analysis...")
    scorer.export_analysis('policy_analysis_results.json')
    
    # Demonstrate detailed analysis
    logger.info("\nGetting detailed analysis for One Big Beautiful Bill Act...")
    detailed = scorer.scoring_history[0]  # First analysis
    
    print("\nDetailed Scoring Breakdown:")
    print("-" * 60)
    for domain, details in detailed['score_details'].items():
        print(f"\n{domain}:")
        print(f"  Positive impacts: {details['positive_score']:.4f}")
        print(f"  Negative impacts: {details['negative_score']:.4f}")
        print(f"  Net impact: {details['net_score']:+.4f}")
        print(f"  Chunks with impact: {details['chunks_with_impact']}/{detailed['chunk_count']}")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == '__main__':
    main()