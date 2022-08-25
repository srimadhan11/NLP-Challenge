## PyTorch Model specific utitily routines

import os

import numpy as np
import torch
import torch.utils.data as tdata
import torch.nn.functional as F

import spacy
from spacy.lang.en import English
from indicnlp.tokenize import indic_tokenize, sentence_tokenize

from helper import iter_batch



# Convert uneven Python List to numpy array using pad_idx as padding element
def pad(arr, pad_idx=0, pad_len=None, dtype=None):
    if pad_len is None: pad_len = max(map(len, arr))
    if dtype   is None: dtype   = np.int64

    np_arr = np.full((len(arr), pad_len), pad_idx, dtype=dtype)
    for n, a in zip(np_arr, arr):
        a = np.array(a, dtype=dtype)
        n[:a.size] = a
    return np_arr


# Compute tokens for hindi sentences
def get_hi_test_tokens(df, hi_lang, sos_tkn, eos_tkn):
    hi_tkns = []
    for hindi in df.get_col('hindi'):
        tkns = [sos_tkn] + indic_tokenize.trivial_tokenize(hindi)  + [eos_tkn]
        hi_tkns.append(hi_lang.tokenize(tkns, add=False))
    return hi_tkns


# Compute tokens for english and hindi sentences
def get_hi_en_tokens(df, hi_lang, en_lang, en_sentencizer, en_lemmatizer, sos_tkn, eos_tkn, add=False):
    hi_tkns, en_tkns = [], []
    for idx,(hindi,english) in enumerate(zip(df.get_col('hindi'), df.get_col('english'))):
        en_doc   = en_sentencizer(english)
        en_sents = tuple(en_doc.sents)

        hi_sents = sentence_tokenize.sentence_split(hindi, lang='hi')
        if len(hi_sents) == len(en_sents):
            for hi_sent in hi_sents:
                tkns = [sos_tkn] + indic_tokenize.trivial_tokenize(hi_sent) + [eos_tkn]
                hi_tkns.append(hi_lang.tokenize(tkns, add=add))
            for en_sent in en_sents:
                tkns = [sos_tkn] + list(token.lemma_.lower() for token in en_lemmatizer(en_sent.text_with_ws)) + [eos_tkn]
                en_tkns.append(en_lang.tokenize(tkns, add=add))
        else:
            tkns = [sos_tkn] + indic_tokenize.trivial_tokenize(hindi)  + [eos_tkn]
            hi_tkns.append(hi_lang.tokenize(tkns, add=add))
            tkns = [sos_tkn] + list(token.lemma_.lower() for token in en_doc) + [eos_tkn]
            en_tkns.append(en_lang.tokenize(tkns, add=add))
    return hi_tkns, en_tkns


# Wrapper class to hold our dataset, which will later used by DataLoader
class TokenizedDataset(tdata.Dataset):
    def __init__(self, src_tkns, tgt_tkns):
        assert src_tkns.shape[0] == tgt_tkns.shape[0]
        self.len = src_tkns.shape[0]
        self.src_tkns = src_tkns
        self.tgt_tkns = tgt_tkns

    def __getitem__(self, index):
        return self.src_tkns[index], self.tgt_tkns[index]

    def __len__(self):
        return self.len


# Run minibatch, and apply backpropagation only if is_train is true
def run_minibatch(src, tgt, model, criterion, device, is_train):
    src = torch.as_tensor(src, device=device, dtype=torch.int64)
    tgt = torch.as_tensor(tgt, device=device, dtype=torch.int64)

    out = model(src, tgt[:, :-1])
    out = F.log_softmax(out, dim=2)
    tgt = tgt[:, 1:]

    out  = out.reshape(-1, out.shape[2])
    tgt  = tgt.reshape(-1)
    loss = criterion(out, tgt)
    if is_train: loss.backward()
    return loss.item()


# Run as single batch or multiple minibatch, based on given minibatch function parameter
def run_batch(src, tgt, model, criterion, device, is_train, minibatch=0):
    if minibatch:
        minibatches = tuple(iter_batch(src.shape[0], minibatch))
        loss = np.empty(len(minibatches), dtype=np.float32)
        for b_idx,(b_start, b_end) in enumerate(minibatches):
            b_src = src[b_start:b_end].reshape(-1, src.shape[1])
            b_tgt = tgt[b_start:b_end].reshape(-1, tgt.shape[1])
            loss[b_idx] = run_minibatch(b_src, b_tgt, model, criterion, device, is_train)
        loss = loss.mean()
    else:
        loss = run_minibatch(src, tgt, model, criterion, device, is_train)
    return loss


# Save model
def save_model(model, path, data_parallel=False):
    module = model.module if data_parallel else model
    torch.save(module.state_dict(), path)
    pass


# Load model
def load_model(model, path, data_parallel=False):
    model.load_state_dict(torch.load(path))
    if data_parallel:
        model = torch.nn.DataParallel(model)
    return model


# Train the model
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, paths, minibatch=0):
    model.train()

    len_train_loader = len(train_loader)
    losses     = np.empty(len_train_loader, dtype=np.float32)
    for idx,(src,tgt) in enumerate(train_loader):
        print(f'Train: {idx}/{len_train_loader}\r', end='')
        optimizer.zero_grad()
        losses[idx] = run_batch(src, tgt, model, criterion, device, is_train=True, minibatch=minibatch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    print(' '*(len(f'Train: {idx}/{len_train_loader}') + 10), end='\r')   # Clear the progress printed with space
    mean_loss = losses.mean()
    scheduler.step(mean_loss)
    return mean_loss


# Test the model on dev set
def dev_epoch(model, dev_loader, criterion, device, minibatch=0):
    model.eval()

    len_dev_loader = len(dev_loader)
    losses    = np.empty(len_dev_loader, dtype=np.float32)
    for idx,(src,tgt) in enumerate(dev_loader):
        print(f'Dev: {idx}/{len_dev_loader}\r', end='')
        losses[idx] = run_batch(src, tgt, model, criterion, device, is_train=False, minibatch=minibatch)
    print(' '*(len(f'Dev: {idx}/{len_dev_loader}') + 10), end='\r')   # Clear the progress printed with space
    mean_loss = losses.mean()
    return mean_loss

