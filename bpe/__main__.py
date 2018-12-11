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

    encoder = Encoder()
    encoder.set_params(pct_bpe=0.3)
    encoder.fit("There is a leader and he is winner")
    text = "There is a leader and he is winner"
    print(encoder.tokenize(text))
    tr = list(encoder.transform(text))
    print(tr)
    print(list(encoder.inverse_transform(tr[0])))


if __name__ == '__main__':
    main(sys.argv[1])
