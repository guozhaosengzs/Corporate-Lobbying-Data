# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2022
# =============================================================================
"""
This module implements helper classes to extract events. The problem
    is formalized as a zero-shot text classification task.

There are two categories of models:
    - Cross-encoder: potentially high acc but low speed (when too many possible events).
    - Bi-encoder: potentially low acc but high speed.
"""

# %%
from typing import List, Union, Any, Dict
from abc import ABC, abstractmethod
from pprint import pprint

import numpy as np

import re
import string
import unicodedata

import torch
from keybert import KeyBERT
from transformers import pipeline


class BaseEventExtractor(ABC):

    @abstractmethod
    def extract(self, texts: Union[List[str], str], events: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class KeyBERTEventExtractor(BaseEventExtractor):

    def __init__(
        self,
        model: str = 'all-mpnet-base-v2',
        top_n_events: int = 4,
        temperature: float = 0.03
    ):
        self._model = KeyBERT(model)
        self._top_n_events = top_n_events
        self._temperature = temperature

    @property
    def model(self):
        return self._model

    @property
    def top_n_events(self):
        return self._top_n_events

    def softmax(self, x: np.ndarray) -> List[float]:
        """Regular softmax function, from scores to probabilities.
            The temperature parameter <self._temperature> is used to
            sharpen the distribution. The smaller the value, the
            sharper the distribution (closer to 'hard max/normal max')"""

        ex = np.exp((x - x.max()) / self._temperature)
        return (ex / ex.sum()).tolist()

    def preprocess(self, s: str) -> str:
        """String pre-processing function, used to reduce noise.
            1. Convert all characters to ASCII
            2. Remove other irrelevant stuff like email address or external url
            3. Remove special symbols like newline character \\n"""

        # Normalize special chars
        s = (unicodedata.normalize('NFKD', s)
                .encode('ascii', 'ignore').decode())

        # Remove irrelevant info
        s = re.sub(r'\S*@\S*\s?', '', s)     # Email
        s = re.sub(r'\S*https?:\S*', '', s)  # URL

        # Keep punctuation and words only
        pattern_keep = (string.punctuation +
                            string.ascii_letters +
                            string.digits +
                            r' ')
        return re.sub(r'[^' + pattern_keep + r']+', '', s)

    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Used to extract most probable events from the given sentence/article
        Args:
            texts (Union[List[str], str]): input sentences such as news article.
            events (List[str]): list of possible events
        Raises:
            ValueError: Invalid article input type.
        Returns:
            List[Dict[str, Any]]: extracted events stored in dictionaries of format:

                {
                    'events': <list of top possible events>,
                    'scores': <The float numbers denote the 'likelihood' of that corresponding event>
                }
        """

        extractor = self._extract_single

        # Direct extraction for single article
        if isinstance(texts, str):
            texts = self.preprocess(texts)
            return [extractor(texts, events)]
        elif isinstance(texts, list):
            texts = [self.preprocess(t) for t in texts]
            return [extractor(t, events) for t in texts]
        else:
            raise ValueError('@ KeyBERTEventExtractor.extract() :: ' +
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')


    def _extract_single(self, text: str, events: List[str] = ['[NULL]']) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        extract, scores = zip(
            *self.model.extract_keywords(
                text, candidates=events, top_n=self.top_n_events
            )
        )
        return {
            'text': text,
            'events': list(extract),
            'scores': self.softmax(np.array(scores)),
            'cosine': list(scores)
        }


class CrossEncoderEventExtractor(BaseEventExtractor):

    def __init__(
        self,
        model: str = 'cross-encoder/nli-MiniLM2-L6-H768',
        top_n_events: int = 4,
        temperature: float = 0.1
    ):
        if torch.cuda.is_available():
            self._model = pipeline('zero-shot-classification', model, batch_size=4, device=0)
        else:
            self._model = pipeline('zero-shot-classification', model)
        self._top_n_events = top_n_events
        self._temperature = temperature

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self.model.device

    @property
    def top_n_events(self):
        return self._top_n_events

    def softmax(self, x: np.ndarray) -> List[float]:
        """Regular softmax function, from scores to probabilities.
            The temperature parameter <self._temperature> is used to
            sharpen the distribution. The smaller the value, the
            sharper the distribution (closer to 'hard max/normal max')"""

        ex = np.exp((x - x.max()) / self._temperature)
        return (ex / ex.sum()).tolist()

    def preprocess(self, s: str) -> str:
        """String pre-processing function, used to reduce noise.
            1. Convert all characters to ASCII
            2. Remove other irrelevant stuff like email address or external url
            3. Remove special symbols like newline character \\n"""

        # Normalize special chars
        s = (unicodedata.normalize('NFKD', s)
                .encode('ascii', 'ignore').decode())

        # Remove irrelevant info
        s = re.sub(r'\S*@\S*\s?', '', s)     # Email
        s = re.sub(r'\S*https?:\S*', '', s)  # URL

        # Keep punctuation and words only
        pattern_keep = (string.punctuation +
                            string.ascii_letters +
                            string.digits +
                            r' ')
        return re.sub(r'[^' + pattern_keep + r']+', '', s)

    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Used to extract most probable events from the given sentence/article
        Args:
            texts (Union[List[str], str]): input sentences such as news article.
            events (List[str]): list of possible events
        Raises:
            ValueError: Invalid article input type.
        Returns:
            List[Dict[str, Any]]: extracted events stored in dictionaries of format:

                {
                    'events': <list of top possible events>,
                    'scores': <The float numbers denote the 'likelihood' of that corresponding event>
                }
        """

        # Direct extraction for single article
        if isinstance(texts, str):
            texts = [self.preprocess(texts)]
            return [self._extract_single(texts, events)]
        elif isinstance(texts, list):
            texts = [self.preprocess(t) for t in texts]
            return self._extract_batch(texts, events)
        else:
            raise ValueError('@ CrossEncoderEventExtractor.extract() :: ' +
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')


    def _extract_single(self, text: str, events: List[str] = ['[NULL]']) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        output = self.model(text, events)
        scores = np.array(output['scores'][:self.top_n_events])
        scores = scores / scores.sum()
        return {
            'text': text,
            'events': output['labels'][:self.top_n_events],
            'scores': self.softmax(scores),
            'cosine': scores.tolist()
        }

    def _extract_batch(self, texts: List[str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Batch processing version of extract single"""

        output = self.model(texts, events)
        for i, o in enumerate(output):
            scores = np.array(o['scores'][:self.top_n_events])
            scores = scores / scores.sum()
            output[i] = {
                'text': o['sequence'],
                'events': o['labels'][:self.top_n_events],
                'scores': self.softmax(scores),
                'cosine': scores.tolist()
            }
        return output

# Sample usage
if __name__ == '__main__':

    texts = ['This is a great movie!', 'This book is so boring. I don\'t like it.', 'I was walking on the street']
    texts = ['This book is so boring. I don\'t like it.']
    labels = ['movie', 'book', 'song', 'went out to run']

    # Alternative
    # extractor = KeyBERTEventExtractor()
    extractor = CrossEncoderEventExtractor()
    pprint(extractor.extract(texts, labels))
