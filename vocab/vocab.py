import string
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import glob
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from collections import Counter
from operator import itemgetter

remove_punctuation_table = str.maketrans('', '', '\'"!.,?:;')
stop_words = set(stopwords.words('english'))

def read_tokens_from_file(file_name):
    with open(file_name) as f:
        doc = f.read()
        # Tokenize
        tokens = word_tokenize(doc)

        # Remove punctuations
        tokens = [w.translate(remove_punctuation_table) for w in tokens]

        tokens = [w.lower() for w in tokens if len(w) > 1 and w.isalpha()]

        # filter out stop words
        tokens = [w for w in tokens if not w in stop_words]

        porter = PorterStemmer()
        tokens = [porter.stem(w) for w in tokens]

        return tokens


if __name__ == "__main__":
    vocab_set = set()
    file_list = glob.glob('./data/train/neg/*.txt')
    file_list = file_list + glob.glob('./data/train/pos/*.txt')

    vocab_counter = Counter()

    for file_name in tqdm(file_list):
        tokens = read_tokens_from_file(file_name)
        vocab_counter.update(tokens)

    vocab_occurrence_list = list(vocab_counter.items())
    vocab_occurrence_list = sorted(vocab_occurrence_list, key=itemgetter(0))
    vocab_occurrence_list = sorted(vocab_occurrence_list, key=itemgetter(1))

    with open('./vocab_counter.txt', 'w') as f:
        for k, c in vocab_occurrence_list:
            print ("%d, %s" % (c, k), file=f)

    min_occurrence = 2
    vocab_list = [k for k, c in vocab_occurrence_list if c >= min_occurrence]
    with open('./vocab.txt', "w") as f:
        for term in  vocab_list:
            print (term, file=f)
