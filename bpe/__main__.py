import json
import sys

from bpe.encoder import Encoder


def main(corpus_path):
    # type: (str) -> None
    """ Loads corpus, learns word and BPE vocab, and writes to stdout.  Assumes corpus is
        line-separated text.
    """
    with open(corpus_path, encoding="utf8") as infile:
        lines = list(map(str.strip, infile))

    encoder = Encoder(silent=True, ngram_max=3)
    encoder.fit(lines)
    print(encoder.tokenize("There is a leader and he is winner"))
    print(json.dumps(encoder.vocabs_to_dict()))


if __name__ == '__main__':
    main(sys.argv[1])
