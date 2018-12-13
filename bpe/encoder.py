# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
import collections
import six

try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator, Union, Set
except ImportError:
    pass

from sklearn.base import TransformerMixin, BaseEstimator
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json

DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'


def is_iterable(arg):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, six.string_types)


class Encoder(BaseEstimator, TransformerMixin):
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, word_tokenizer=None,
                 silent=True, required_tokens=None,
                 strict=False):

        self.EOW = DEFAULT_EOW
        self.SOW = DEFAULT_SOW
        self.eow_len = len(self.EOW)
        self.sow_len = len(self.SOW)
        self.required_tokens = list(set(required_tokens or []))
        self.vocab_size = 8192
        self.pct_bpe = 0.2
        self.word_vocab_size = max([int(self.vocab_size * (1 - self.pct_bpe)),
                                    len(self.required_tokens or [])])
        self.bpe_vocab_size = self.vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab_words = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = 2
        self.ngram_max = 2
        self.word_ngram_min = 2
        self.word_ngram_max = 2
        self.strict = strict
        self.tokenize_on_word_ngrams = False
        self.tokenize_symbols = True

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise ValueError("Encoder does not have parameter '{}'".format(parameter))
            if parameter == "vocab_size":
                if value < 1:
                    raise ValueError('vocab size must be greater than 0.')
            if parameter == "SOW":
                self.sow_len = len(value)
            if parameter == "EOW":
                self.eow_len = len(value)
            if parameter == "tokenize_symbols":
                if not value:
                    self.word_vocab_size = self.vocab_size
                    self.bpe_vocab_size = 0
            if parameter == "vocab_size":
                self.word_vocab_size = max([int(value * (1 - self.pct_bpe)),
                                            len(self.required_tokens or [])])
                self.bpe_vocab_size = value - self.word_vocab_size
            if parameter == "pct_bpe":
                self.word_vocab_size = max([int(self.vocab_size * (1 - value)),
                                            len(self.required_tokens or [])])
                self.bpe_vocab_size = self.vocab_size - self.word_vocab_size
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"vocab_size": self.vocab_size, "pct_bpe": self.pct_bpe,
                "ngram_min": self.ngram_min, "ngram_max": self.ngram_max,
                "word_ngram_min": self.word_ngram_min, "word_ngram_max": self.word_ngram_max,
                "tokenize_symbols": self.tokenize_symbols,
                "EOW": self.EOW, "SOW": self.SOW, }

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    def byte_pair_counts(self, words):
        # type: (Encoder, Iterable[str]) -> Iterable[collections.Counter]
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = collections.Counter()  # type: collections.Counter
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def word_pair_counts(self, line):
        # type: (Encoder, str) -> Iterable[collections.Counter]
        """ Counts space separated token character pairs:
            [('This is sparta', 4}] -> {'This is': 4, 'is sparta': 4}
        """
        words = self.word_tokenizer(line)
        for n in range(self.word_ngram_min, self.word_ngram_max + 1):
            bp_counts = collections.Counter()  # type: collections.Counter
            sld_wnd = list(toolz.sliding_window(n, range(len(words))))
            sln_wnd_txt = [words[pair[0]: pair[len(pair) - 1] + 1] for pair in sld_wnd]
            for word_ngram in sln_wnd_txt:
                bp_counts[" ".join(word_ngram)] = 1
            yield bp_counts

    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = collections.Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Union[str,Iterable[str]]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        if is_iterable(sentences):
            word_counts = collections.Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))
        else:
            lst = self.word_tokenizer(sentences)
            word_counts = collections.Counter(word for word in lst)
        for token in set(self.required_tokens or []):
            word_counts[token] = int(2**63)
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = collections.Counter()  # type: collections.Counter
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def learn_bpe_vocab_words(self, lines):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = collections.Counter()  # type: collections.Counter
        for line in lines:
            for idx, byte_pair_count in enumerate(self.word_pair_counts(line)):
                for byte_pair, count in byte_pair_count.items():
                    vocab[byte_pair] += count

                if (idx + 1) % 10000 == 0:
                    self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.word_vocab_size]
        return {bp: idx for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text):
        # type: (Encoder, Union[Iterable[str], str]) -> None
        if is_iterable(text):
            self.tokenize_on_word_ngrams = True
            """ Learn vocab from text. """
            _text = [l.strip() for l in text]

            self.word_vocab = self.learn_bpe_vocab_words(_text)
            self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}

            if self.tokenize_symbols:
                remaining_words = self.find_remaining_words(self.word_vocab, _text)
                self.bpe_vocab = self.learn_bpe_vocab(remaining_words)
                self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}
        else:
            self.tokenize_on_word_ngrams = False
            """ Learn vocab from text. """
            _text = text.strip()

            # First, learn word vocab
            self.word_vocab = self.learn_word_vocab(_text)
            self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}

            if self.tokenize_symbols:
                remaining_words = [word for word in self.word_tokenizer(_text)
                                   if word not in self.word_vocab]
                self.bpe_vocab = self.learn_bpe_vocab(remaining_words)
                self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        # type: (int, Dict[str, int]) -> None
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def subline_tokenize(self, line):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        words = self.word_tokenizer(line)
        end_idx = min([len(words), self.word_ngram_max])
        sw_tokens = []
        start_idx = 0

        while start_idx < len(words):
            subword_list = words[start_idx:end_idx]
            subword = " ".join(subword_list)
            if subword in self.word_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(words), start_idx + self.word_ngram_max])
            elif len(subword_list) == 1:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(words), start_idx + self.word_ngram_max])
            else:
                end_idx -= 1

        return sw_tokens

    def _tokenize(self, sentence):
        # type: (Encoder, str) -> List[str]
        if not self.tokenize_on_word_ngrams:
            """ Split a sentence into word and subword tokens """
            word_tokens = self.word_tokenizer(sentence.strip())

            tokens = []
            for word_token in word_tokens:
                if word_token in self.word_vocab:
                    tokens.append(word_token)
                else:
                    if self.tokenize_symbols:
                        tokens.extend(self.subword_tokenize(word_token))
                    else:
                        tokens.append(word_token)

            return tokens
        else:
            result = []
            tokens = self.subline_tokenize(sentence)
            for token in tokens:
                if token in self.word_vocab:
                    result.append(token)
                else:
                    if self.tokenize_symbols:
                        result += self.subword_tokenize(token)
                    else:
                        result.append(token)

            return result

    def transform(self, sentences):
        return self._tokenize(sentences)

    def vocabs_to_dict(self, dont_warn=False):
        # type: (Encoder, bool) -> Dict[str, Dict[str, int]]
        """ Turns vocab into dict that is json-serializeable """
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer.  You'll need to reassign it when you load the "
                  "model!")
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
            'word_ngrams': self.bpe_vocab_words,
            "tokenize_on_word_ngrams": self.tokenize_on_word_ngrams,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'word_ngram_min': self.word_ngram_min,
                'word_ngram_max': self.word_ngram_max,
                'required_tokens': self.required_tokens,
                'strict': self.strict,
                'EOW': self.EOW,
                'SOW': self.SOW,
            }
        }

    def save(self, outpath, dont_warn=False):
        # type: (Encoder, str, bool) -> None
        """ Serializes and saves encoder to provided path """
        with open(outpath, 'w') as outfile:
            json.dump(self.vocabs_to_dict(dont_warn), outfile)

    @classmethod
    def from_dict(cls, vocabs):
        # type: (Any, Dict[str, Dict[str, int]]) -> Encoder
        """ Load encoder from dict produced with vocabs_to_dict """
        encoder = Encoder(**vocabs['kwargs'])
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']
        encoder.bpe_vocab_words = vocabs['word_ngrams']
        encoder.tokenize_on_word_ngrams = vocabs['tokenize_on_word_ngrams']

        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}

        return encoder

    @classmethod
    def load(cls, in_path):
        # type: (Any, str) -> Encoder
        """ Loads an encoder from path saved with save """
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)

    def find_remaining_words(self, word_vocab, _text):
        # type: (Encoder, Dict[str, int], List[str]) -> Set[str]
        tokens = set()
        for line in _text:
            word_tokens = self.subline_tokenize(line)
            for token in word_tokens:
                if token not in word_vocab:
                    tokens.add(token)
        return tokens
