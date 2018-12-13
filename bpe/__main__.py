import json
import sys

from bpe.encoder import Encoder


def main(corpus_path):
    # type: (str) -> None
    """ Loads corpus, learns word and BPE vocab, and writes to stdout.  Assumes corpus is
        line-separated text.
    """
    train_text = [
        "Hello world from earth",
        "Hello George who has two ears",
        "Hello Georgia which has a big capital city",
        "A litte George said hello",
    ]

    train_text_single = "Hello world from earth. Hello George who has two ears. Hello Georgia which has a big capital city. A litte George said hello. Linking ideas in domain is a good idea"

    encoder = Encoder()
    encoder.set_params(tokenize_symbols=True, vocab_size=30, pct_bpe=1, ngram_min=2, ngram_max=5)
    encoder.fit(train_text_single)
    text = "Say Hello to guys from Georgia"
    print(encoder.transform(text))
    print(encoder.bpe_vocab)
    print(encoder.word_vocab)


if __name__ == '__main__':
    main(sys.argv[1])
