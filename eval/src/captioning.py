from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
import nltk
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge as RougeC
from pycocoevalcap.cider.cider import Cider

# Class to calculate caption evaluation metrics using the pycocoevalcap library
class Caption_Scorer():
    def __init__(self, ref, gt):
        """
        Initialize the Caption_Scorer with reference and generated captions.
        Sets up the evaluation metrics for BLEU, ROUGE_L, and CIDEr.
        """
        self.ref = ref  # Reference captions (ground truth)
        self.gt = gt  # Generated captions (predicted)
        print('Setting up scorers...')
        
        # Define the list of metrics to be calculated
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),  # BLEU scores for 1-4 n-grams
            # (Meteor(), "METEOR"), # Requires Java version 11.0.16+ (commented out for compatibility)
            (RougeC(), 'ROUGE_L'),  # ROUGE-L score
            (Cider(), 'CIDEr'),  # CIDEr score
            # (Spice(), "SPICE"), # Requires Java version 11.0.16+ (commented out for compatibility)
        ]

    def compute_scores(self):
        """
        Compute evaluation scores for all metrics in the scorers list.
        Returns a dictionary of scores for each metric.
        """
        total_scores = {}  # Dictionary to store scores for each metric
        
        # Iterate over the scorers and calculate scores
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)  # Calculate scores
            if isinstance(method, list):
                # For BLEU scores (multiple methods), store all results in a list
                total_scores['Bleu'] = [x * 100 for x in score]  # Convert to percentage
            else:
                # For other metrics, store the score as a single value
                total_scores[method] = score * 100  # Convert to percentage

        print('*****DONE*****')
        return total_scores  # Return the computed scores


def calculate_bleu_scores(reference_captions, generated_captions):
    """
    Calculate BLEU scores using NLTK for 1-gram to 4-gram precision.
    """
    # Tokenize the reference and generated captions
    reference_list = [[nltk.word_tokenize(ref) for ref in refs] for refs in reference_captions]
    generated_list = [nltk.word_tokenize(gen) for gen in generated_captions]

    # Compute BLEU scores for different n-gram weights
    bleu1 = corpus_bleu(reference_list, generated_list, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(reference_list, generated_list, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(reference_list, generated_list, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(reference_list, generated_list, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def calculate_meteor_scores(reference_captions, generated_captions):
    """
    Calculate METEOR scores for the given reference and generated captions.
    """
    # Compute METEOR score for each pair of reference and generated captions
    meteor_scores = [single_meteor_score(nltk.word_tokenize(ref), nltk.word_tokenize(pred)) for ref, pred in zip(reference_captions, generated_captions)]
    meteor_avg = sum(meteor_scores) / len(meteor_scores)  # Calculate average METEOR score
    return meteor_avg


def calculate_rouge_scores(reference_captions, generated_captions):
    """
    Calculate ROUGE scores for the given reference and generated captions.
    """
    rouge = Rouge()  # Initialize the ROUGE scorer
    # Compute ROUGE scores (averaged over all captions)
    scores = rouge.get_scores(generated_captions, reference_captions, avg=True, ignore_empty=True)
    return scores


def evaluate_captioning(reference_captions, generated_captions):
    """
    Perform overall evaluation of generated captions against reference captions.
    Computes scores for BLEU, METEOR, ROUGE-L, and CIDEr metrics.
    """
    # Compute METEOR scores
    meteor_score = calculate_meteor_scores(reference_captions, generated_captions)
    
    # Compute ROUGE scores (currently has a known bug, left as a placeholder)
    rouge_method = calculate_rouge_scores(reference_captions, generated_captions)

    # Prepare data for pycocoevalcap metrics (BLEU, ROUGE-L, CIDEr)
    ref, gt = evaluate_captioning2(reference_captions, generated_captions)
    
    # Initialize the Caption_Scorer with the prepared data
    scorer = Caption_Scorer(ref, gt)
    
    # Compute all pycocoevalcap scores
    total_score = scorer.compute_scores()

    # Return individual scores for each metric
    return total_score['Bleu'], meteor_score, rouge_method, total_score['ROUGE_L'], total_score['CIDEr']


def evaluate_captioning2(reference_captions, generated_captions):
    """
    Prepare captions for evaluation by pycocoevalcap.
    Formats the reference and generated captions into dictionaries indexed by ID.
    """
    ref = {}  # Dictionary to store reference captions
    gt = {}  # Dictionary to store generated captions

    # Populate the dictionaries with captions, indexed by ID
    for i, (ref_data, gt_data) in enumerate(zip(reference_captions, generated_captions)):
        ref[str(i)] = [ref_data]  # Reference captions can have multiple sentences
        gt[str(i)] = [gt_data]  # Generated captions

    return ref, gt  # Return formatted reference and generated captions
