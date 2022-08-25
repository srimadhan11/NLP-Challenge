## Common helper routines

import os
import csv
import difflib
import itertools

import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score


def cuda_blocking():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    pass


def mkdir_p(dirname):
    if os.path.exists(dirname):
        print(f'I: Path "{dirname}" already exist')
    else:
        os.mkdir(dirname)
    pass


# Refer: https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
def diff(a, b):
    matcher = difflib.SequenceMatcher(None, a, b)
    def process_opcode(tag, i1, i2, j1, j2):
        if tag == 'equal'  : return         matcher.a[i1:i2]
        if tag == 'insert' : return '{+ ' + matcher.b[j1:j2] + '}'
        if tag == 'delete' : return '{- ' + matcher.a[i1:i2] + '}'
        if tag == 'replace': return '{'   + matcher.a[i1:i2] + ' -> ' + matcher.b[j1:j2] + '}'
        assert False, "Unknown tag: {}".format(tag)
    return ''.join(process_opcode(*opcode) for opcode in matcher.get_opcodes())


def compute_metrics(references, hypotheses, is_files=False, verbose=True):
    if is_files:
        with open(references, 'r'), open(hypotheses, 'r') as ref_file, hyp_file:
            references = ref_file.readlines()
            hypotheses = hyp_file.readlines()

    assert len(references) == len(hypotheses)

    bleu_score, meteor_score = 0, 0
    total_num = len(references)
    for ref,hyp in zip(references, hypotheses):
        bleu_score   += sentence_bleu([ref.split(" ")], hyp.split(" "))
        meteor_score += single_meteor_score(ref, hyp)

    bleu_result   = bleu_score   / total_num
    meteor_result = meteor_score / total_num

    if verbose:
        print(f"bleu score: {bleu_result}, meteor score: {meteor_result}")
    return bleu_result, meteor_result


# Function that gives start and end indices of batch as generator object
def iter_batch(data_len, batch_size):
    cur_idx, end_idx = 0, batch_size
    while end_idx < data_len:
        yield cur_idx, end_idx
        cur_idx, end_idx = end_idx, end_idx+batch_size
    yield cur_idx, data_len


# To split the train and dev set
def split_df(df, split=0.8):
    assert 0 < split < 1
    split_mask = np.random.rand(df.no_of_row()) < split
    return df.mask_row(split_mask), df.mask_row(~split_mask)         # train_df, dev_df


# Helper class to parse and extract information from csv file
class csv_data:
    def __init__(self, filename=None, cols=None, data=None):
        if data is None:
            assert filename is not None and cols is not None
            data = dict()
            for col in cols:
                data[col] = list()

            with open(filename, newline='') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                for row in csv_reader:
                    for col in cols:
                        data[col].append(row[col])
        self.data = data
        pass
    
    def get(self, row, col):
        return self.get_col(col)[row]
    
    def get_col(self, col):
        return self.data[col]
    
    def get_row(self, row):
        return [self.data[col][row] for col in self.data]
    
    def no_of_col(self):
        return len(self.data)
    
    def no_of_row(self):
        key0 = next(iter(self.data))
        col0_len = len(self.data[key0])
        assert all(len(self.data[col]) == col0_len for col in self.data)
        return col0_len
    
    def shape(self):
        return [self.no_of_row(), self.no_of_col()]
    
    def mask_row(self, mask):
        data = dict()
        for col in self.data:
            data[col] = list()
        
        for row,m in enumerate(mask):
            if not m: continue
            for col in self.data:
                data[col].append(self.get(row, col))
        return csv_data(data=data)
    pass
