import numpy as np
import os
import json, time, copy
import math
import pdb

from tqdm import tqdm
import random
import pickle
from datetime import datetime
from pytz import timezone
from word2number import w2n
import string, re
from collections import Counter, defaultdict
from pprint import pprint
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])
np.set_printoptions(precision=4)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gt_file", type=str, default="/data1/abhiram/retvqa/final_retvqa_v4/train_val_test_v4.json") #
parser.add_argument("--result_file", type=str, default="/data1/abhiram/webqa/VL-T5/VL-T5/snap/retvqa/VLBart/test_results_retvqa_v4_50_negs.json")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
args = parser.parse_args()

import sys
sys.path.append("/data1/abhiram/retvqa/final_retvqa_v2/BARTScore")
from bart_score import BARTScorer

bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
bart_scorer_ParaBank.load(path='/data1/abhiram/retvqa/final_retvqa_v2/BARTScore/weights/bart.pth') # Please change the path to bart.pth


def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str((w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a, exclude=""):
    if len(cands) == 0: return 0
    bow_a = normalize_text(a).split()
    RE = []
    e = normalize_text(exclude).split()
    for c in cands:
        bow_c = [w for w in normalize_text(c).split() if not w in e]
        
        common = Counter(bow_c) & Counter(bow_a)
        num_same = sum(common.values())
        
        if num_same == 0:
            return (0)

        recall = 1.0 * num_same / len(bow_a)
        RE.append(recall)
        
        
    RE_avg = np.mean(RE)
    return (RE_avg)


TABLE = str.maketrans(dict.fromkeys(string.punctuation)) 
def normalize_text_for_bart(x): # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())

def compute_bartscore_ParaBank(c, a, switch=False):
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch: score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else: score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score


original_data = json.load(open(args.gt_file, 'r'))
answer_data = json.load(open(args.result_file, 'r'))
fluency_scores = []
recall_scores = []
fa_scores = []

fails = 0

if args.end == -1:
    args.end = len(list(answer_data.keys()))

# calculate accuracy
accuracy_scores = []
fluency_scores = []
fa_scores = []

q_types = []

for key, value in original_data.items():
    q_types.append(value['q_type'])

q_types = list(set(q_types))

for i in range(len(q_types)):
    accuracy_scores.append([])
    fluency_scores.append([])
    fa_scores.append([])


binary_accuracy = []
binary_fluency = []
binary_fa = []

generative_accuracy = []
generative_fluency = []
generative_fa = []


count = 0
overall_accuracy = []
overall_fluency = []
overall_fa = []

# nvcr.io/nvidia/pytorch:22.06-py3

# number_of_keys = list(original_data.keys())

for index, (key, value) in enumerate(tqdm(list(answer_data.items())[args.start:args.end])):
    # if index < args.start:
    #     continue
    try:
        # if list(key)[0] in generative_answers:
            original_answer = original_data[key]['answer'].lower()
            precise_answer = original_data[key]['precise_answer'].lower()
            # print(original_answer, value)
            normalizer = compute_bartscore_ParaBank([original_answer], [original_answer])
            fluency_score = min(1, np.max(compute_bartscore_ParaBank([value]*len([original_answer]), [original_answer])/np.array(normalizer)))
            accuracy = compute_vqa_metrics([value.lower()], precise_answer, "")

            q_type = original_data[key]['q_type']
            q_type_index = q_types.index(q_type)

            accuracy_scores[q_type_index].append(accuracy)
            fluency_scores[q_type_index].append(fluency_score)
            fa_scores[q_type_index].append(accuracy*fluency_score)

            overall_accuracy.append(accuracy)
            overall_fluency.append(fluency_score)
            overall_fa.append(accuracy*fluency_score)

            if original_data[key]['a_type'] == 'binary':
                binary_accuracy.append(accuracy)
                binary_fluency.append(fluency_score)
                binary_fa.append(accuracy*fluency_score)
            elif original_data[key]['a_type'] == 'generative':
                generative_accuracy.append(accuracy)
                generative_fluency.append(fluency_score)
                generative_fa.append(accuracy*fluency_score)

            if (count+1) % 100 == 0:
                # print(f'Overall Fluency score: {sum(overall_accuracy)/number_of_keys:0.4f}')
                # print(f'Overall Recall score: {sum(overall_fluency)/number_of_keys:0.4f}')
                # print(f'Overall FA score: {sum(overall_fa)/number_of_keys:0.4f}')

                print(f'Overall Fluency score: ', np.mean(overall_fluency))
                print(f'Overall Recall score: ', np.mean(overall_accuracy))
                print(f'Overall FA score: ', np.mean(overall_fa))
                print('---------------------\n\n')

            count += 1  
        
    except:
        fails += 1
        continue

    # if index == 50:
    #     break

print(f'Fails: {fails}')
for id, key in enumerate(q_types):
    value = id
    # print(f'Fluency score for {key}: {sum(accuracy_scores[value])/number_of_keys:0.4f}')
    # print(f'Recall score for {key}: {sum(fluency_scores[value])/number_of_keys:0.4f}')
    # print(f'FA score for {key}: {sum(fa_scores[value])/number_of_keys:0.4f}')

    print(f'Fluency score for {key}: ', np.mean(fluency_scores[value]))
    print(f'Recall score for {key}: ', np.mean(accuracy_scores[value]))
    print(f'FA score for {key}: ', np.mean(fa_scores[value]))

# print(f'Overall Fluency score: {sum(overall_accuracy)/number_of_keys:0.4f}')
# print(f'Overall Recall score: {sum(overall_fluency)/number_of_keys:0.4f}')
# print(f'Overall FA score: {sum(overall_fa)/number_of_keys:0.4f}')

print(f'Overall Fluency score: ', np.mean(overall_fluency))
print(f'Overall Recall score: ', np.mean(overall_accuracy))
print(f'Overall FA score: ', np.mean(overall_fa))
print('---------------------\n\n')

# print(f'Binary Fluency score: {sum(binary_accuracy)/number_of_keys:0.4f}')
# print(f'Binary Recall score: {sum(binary_fluency)/number_of_keys:0.4f}')
# print(f'Binary FA score: {sum(binary_fa)/number_of_keys:0.4f}')

print(f'Binary Fluency score: ', np.mean(binary_fluency))
print(f'Binary Recall score: ', np.mean(binary_accuracy))
print(f'Binary FA score: ', np.mean(binary_fa))
print('---------------------\n\n')


# print(f'Generative Fluency score: {sum(generative_accuracy)/number_of_keys:0.4f}')
# print(f'Generative Recall score: {sum(generative_fluency)/number_of_keys:0.4f}')
# print(f'Generative FA score: {sum(generative_fa)/number_of_keys:0.4f}')

print(f'Generative Fluency score: ', np.mean(generative_fluency))
print(f'Generative Recall score: ', np.mean(generative_accuracy))
print(f'Generative FA score: ', np.mean(generative_fa))
