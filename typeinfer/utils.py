import collections
import math

import logging
import sys
from pathlib import Path

def init_log(file=None, level=logging.INFO):
    format = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.getLogger().setLevel(level)
    formatter = logging.Formatter(format)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    stderr.setLevel(level)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(stderr)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=file, mode="w", encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)

class Bleu:
    @staticmethod
    def get_ngrams(segment, max_order):
        """
            Extracts all n-grams upto a given maximum order from an input segment.
            Args:
                segment: text segment from which n-grams will be extracted.
                max_order: maximum length in tokens of the n-grams returned by this methods.
            Returns:
                The Counter containing all n-grams upto max_order in segment
                with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    @staticmethod
    def compute_bleu(references, candidates, max_order=4, smooth=False):
        """
        Computes BLEU score of translated segments against one or more references.
        Args:
            references: list of references for each candidate. Each reference should be tokenized into a list of tokens.
            candidates: list of candidates to score. Each candidate should be tokenized into a list of tokens.
            max_order: Maximum n-gram order to use when computing BLEU score.
            smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
            3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        length = len(references)
        reference_length = 0
        candidate_length = 0
        for i in range(length):
            reference_length += len(references[i])
            candidate_length += len(candidates[i])
            reference_ngram_counts = Bleu.get_ngrams(references[i],max_order)
            candidate_ngram_counts = Bleu.get_ngrams(candidates[i],max_order)
            overlap = reference_ngram_counts & candidate_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1,max_order+1):
                possible_matches = len(candidates[i]) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0,max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(candidate_length) / reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp
        return (bleu, precisions, bp, ratio, candidate_length, reference_length)