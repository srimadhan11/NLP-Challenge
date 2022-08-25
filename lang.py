import pickle

import numpy as np

from utils import pad, get_hi_en_tokens, get_hi_test_tokens


class Lang:
    '''A class to handle all the vocabulary related routines.'''
    def __init__(self, name, special_tokens=None):
        self.name       = name
        self.n_words    = 0
        self.word2index = {}
        self.index2word = {}
        if special_tokens is None:
            special_tokens = ("[SOS]", "[EOS]", "[PAD]", "[UNK]")
        self.tokenize(special_tokens, add=True)
        pass

    def tokenize(self, obj, add=False):
        func = self.__add_tokenize if add else self.__only_tokenize
        return tuple(func(word) for word in obj) if isinstance(obj, (list, tuple)) else func(obj)

    def __add_tokenize(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        return self.word2index[word]

    def __only_tokenize(self, word):
        return self.word2index[word if word in self.word2index else '[UNK]']

    def sentencize(self, obj):
        if isinstance(obj, (list, tuple)):
            return tuple(self.index2word[idx] for idx in obj)
        return self.index2word[obj]

    def __len__(self):
        return self.n_words

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2index[key]
        if isinstance(key, int):
            return self.index2word[key]
        raise Exception("Unknown key type: {}".format(key))

    def __contains__(self, memb):
        if isinstance(memb, str):
            return memb in self.word2index
        if isinstance(key, int):
            return memb in self.index2word
        raise Exception("Unknown member type: {}".format(memb))


# Compute test tokens
def compute_test(test, hi_lang):
    sos_tkn, eos_tkn = "[SOS]", "[EOS]"
    hi_tkns = get_hi_test_tokens(test, hi_lang, sos_tkn, eos_tkn)
    hi_tkns = pad(hi_tkns, hi_lang['[PAD]'], dtype=np.int32)
    return hi_tkns


# Load Lang object and (train+dev) tokens
def load_lang_and_data(paths):
    hi_tkns     = np.load(paths('hi_tkns'))
    en_tkns     = np.load(paths('en_tkns'))
    dev_hi_tkns = np.load(paths('dev_hi_tkns'))
    dev_en_tkns = np.load(paths('dev_en_tkns'))

    with open(paths('split_train'), 'rb') as pkl_file:
        train = pickle.load(pkl_file)
    with open(paths('split_dev'  ), 'rb') as pkl_file:
        dev = pickle.load(pkl_file)
    with open(paths('hi_lang'), 'rb') as pkl_file:
        hi_lang = pickle.load(pkl_file)
    with open(paths('en_lang'), 'rb') as pkl_file:
        en_lang = pickle.load(pkl_file)
    return (train, dev), (hi_tkns, en_tkns), (dev_hi_tkns, dev_en_tkns), (hi_lang, en_lang)


# Compute Lang object and (train+dev) tokens
def compute_lang_and_data(train, dev, en_sentencizer, en_lemmatizer):
    sos_tkn, eos_tkn = "[SOS]", "[EOS]"
    hi_lang, en_lang = Lang('hindi'), Lang('english')

    # train tokens
    hi_tkns, en_tkns = get_hi_en_tokens(train, hi_lang, en_lang, en_sentencizer, en_lemmatizer, sos_tkn, eos_tkn, add=True)
    hi_tkns = pad(hi_tkns, hi_lang['[PAD]'], dtype=np.int32)
    en_tkns = pad(en_tkns, en_lang['[PAD]'], dtype=np.int32)

    # test tokens
    dev_hi_tkns, dev_en_tkns = get_hi_en_tokens(dev, hi_lang, en_lang, en_sentencizer, en_lemmatizer, sos_tkn, eos_tkn, add=False)
    dev_hi_tkns = pad(dev_hi_tkns, hi_lang['[PAD]'], dtype=np.int32)
    dev_en_tkns = pad(dev_en_tkns, en_lang['[PAD]'], dtype=np.int32)
    return (train, dev), (hi_tkns, en_tkns), (dev_hi_tkns, dev_en_tkns), (hi_lang, en_lang)


# Save Lang object and (train+dev) tokens
def save_lang_and_data(train, dev, hi_tkns, en_tkns, dev_hi_tkns, dev_en_tkns, hi_lang, en_lang, paths):
    np.save(paths('hi_tkns'), hi_tkns)
    np.save(paths('en_tkns'), en_tkns)
    np.save(paths('dev_hi_tkns'), dev_hi_tkns)
    np.save(paths('dev_en_tkns'), dev_en_tkns)

    with open(paths('split_train'), 'wb') as pkl_file:
        pickle.dump(train, pkl_file)
    with open(paths('split_dev'  ), 'wb') as pkl_file:
        pickle.dump(dev  , pkl_file)
    with open(paths('hi_lang'), 'wb') as pkl_file:
        pickle.dump(hi_lang, pkl_file)
    with open(paths('en_lang'), 'wb') as pkl_file:
        pickle.dump(en_lang, pkl_file)
    pass

