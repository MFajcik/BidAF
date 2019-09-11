import csv
import json
import os
import shutil
import time
import string
import sys
from urllib import request
import numpy as np
import nltk
import torchtext.data as data
from typing import List, Tuple, Dict
import logging

from torchtext.data import RawField, Example

from tokenizers.spacy_tokenizer import tokenize, char_span_to_token_span, tokenize_and_join

TRAIN_V1_URL = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/train-v1.1.json'
DEV_V1_URL = 'https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json'
TRAIN = "train-v1.1.json"
VALIDATION = "dev-v1.1.json"


def download_url(path, url):
    sys.stderr.write(f'Downloading from {url} into {path}\n')
    sys.stderr.flush()
    request.urlretrieve(url, path)


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


JOIN_TOKEN = "█"


class SquadDataset(data.Dataset):

    def __init__(self, data, fields: List[Tuple[str, data.Field]], cachedir='.data/squad', **kwargs):
        f = os.path.join(cachedir, data)
        preprocessed_f = f + "_preprocessed.json"
        if not os.path.exists(preprocessed_f):
            s_time = time.time()
            raw_examples = SquadDataset.get_example_list(f)
            self.save(preprocessed_f, raw_examples)
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time}s")

        s_time = time.time()
        examples = self.load(preprocessed_f, fields)
        logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")

        super(SquadDataset, self).__init__(examples, fields, **kwargs)

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return [data.Example.fromlist([
                e["id"],
                e["topic"],
                e["paragraph_token_positions"],
                e["raw_paragraph_context"],
                e["paragraph_context"],
                e["paragraph_context"],
                e["paragraph_context"],
                e["question"],
                e["question"],
                e["question"],
                e["a_start"],
                e["a_end"],
                e["a_extracted"],
                e["a_gt"]
            ], fields) for e in raw_examples]

    @classmethod
    def splits(cls, fields, cachedir='.data/squad'):
        cls.check_for_download(cachedir)
        train_data = cls(TRAIN, fields, cachedir=cachedir)
        val_data = cls(VALIDATION, fields, cachedir=cachedir)
        return tuple(d for d in (train_data, val_data)
                     if d is not None)

    @staticmethod
    def check_for_download(cachedir):
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
            try:
                download_url(os.path.join(cachedir, TRAIN), TRAIN_V1_URL)
                download_url(os.path.join(cachedir, VALIDATION), DEV_V1_URL)
            except BaseException as e:
                sys.stderr.write(f'Download failed, removing directory {cachedir}\n')
                sys.stderr.flush()
                shutil.rmtree(cachedir)
                raise e

    @staticmethod
    def get_example_list(file):
        examples = []
        cnt = 0

        ## debug
        f = open(f".data/squad/debug_{os.path.basename(file)}.csv", "a+")
        problems = 0

        with open(file) as fd:
            data_json = json.load(fd)
            for data_topic in data_json["data"]:
                topic_title = data_topic["title"]
                for paragraph in data_topic["paragraphs"]:
                    paragraph_tokens, paragraph_context = tokenize(paragraph["context"])
                    paragraph_token_positions = [[token.idx, token.idx + len(token.text)] for token in paragraph_tokens]
                    joined_paragraph_context = JOIN_TOKEN.join(paragraph_context)
                    for question_and_answers in paragraph['qas']:
                        example_id = question_and_answers["id"]
                        question = tokenize_and_join(question_and_answers['question'])
                        answers = question_and_answers['answers']

                        for possible_answer in answers:
                            answer_start_ch = possible_answer["answer_start"]
                            answer_end = possible_answer["answer_start"] + len(possible_answer["text"])
                            answer_tokens, answer = tokenize(possible_answer["text"])

                            answer_locations = find_sub_list(answer, paragraph_context)
                            if len(answer_locations) > 1:
                                # get start character offset of each span
                                answer_ch_starts = [paragraph_tokens[token_span[0]].idx for token_span in
                                                    answer_locations]
                                distance_from_gt = np.abs((np.array(answer_ch_starts) - answer_start_ch))
                                closest_match = distance_from_gt.argmin()

                                answer_start, answer_end = answer_locations[closest_match]
                            elif not answer_locations:
                                # Call heuristic from AllenNLP to help :(
                                token_span = char_span_to_token_span(
                                    [(t.idx, t.idx + len(t.text)) for t in paragraph_tokens],
                                    (answer_start_ch, answer_end))
                                answer_start, answer_end = token_span[0]
                            else:
                                answer_start, answer_end = answer_locations[0]
                            cnt += 1

                            ## Debug
                            def is_correct():
                                def remove_ws(s):
                                    return "".join(s.split())

                                csvf = csv.writer(f, delimiter=',')
                                if remove_ws(possible_answer["text"]) != remove_ws(
                                        "".join(paragraph_context[answer_start:answer_end + 1])):
                                    csvf.writerow({"id": example_id,
                                                   "topic": topic_title,
                                                   "raw_paragraph_context": paragraph["context"],
                                                   "paragraph_context": joined_paragraph_context,
                                                   "paragraph_token_positions": paragraph_token_positions,
                                                   "question": question,
                                                   "a_start": answer_start,
                                                   "a_end": answer_end,
                                                   "a_extracted": JOIN_TOKEN.join(
                                                       paragraph_context[answer_start:answer_end + 1]),
                                                   "a_gt": possible_answer["text"]}.values())
                                    return False
                                return True

                            if not is_correct():
                                problems += 1

                            examples.append({"id": example_id,
                                             "topic": topic_title,
                                             "raw_paragraph_context": paragraph["context"],
                                             "paragraph_context": joined_paragraph_context,
                                             "paragraph_token_positions": paragraph_token_positions,
                                             "question": question,
                                             "a_start": answer_start,
                                             "a_end": answer_end,
                                             "a_extracted": JOIN_TOKEN.join(
                                                 paragraph_context[answer_start:answer_end + 1]),
                                             "a_gt": possible_answer["text"]})

            # debug
            logging.info(f"# problems: {problems}")
            logging.info(f"Preprocessing problems affect {problems / len(examples) / 100:.6f} % of the dataset.")
            return examples

    @staticmethod
    def prepare_fields():
        WORD_field = data.Field(batch_first=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), lower=True)
        return [
            ('id', data.RawField()),
            ('topic_title', data.RawField()),
            ('document_token_positions', data.RawField()),
            ('raw_document_context', data.RawField()),
            ('document', WORD_field),
            ('document_char', data.RawField()),
            ('raw_document', data.RawField()),
            ('question', WORD_field),
            ('question_char', data.RawField()),
            ('raw_question', data.RawField()),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('ext_answer', data.RawField()),
            ('gt_answer', data.RawField())
        ]

    @staticmethod
    def prepare_fields_char():
        WORD_field = data.Field(batch_first=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), lower=True,
                                include_lengths=True)
        CHAR_field = data.Field(batch_first=True, tokenize=list, lower=True)
        CHAR_nested_field = data.NestedField(CHAR_field, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN))
        return [
            ('id', data.RawField()),
            ('topic_title', data.RawField()),
            ('document_token_positions', data.RawField()),
            ('raw_document_context', data.RawField()),
            ('document', WORD_field),
            ('document_char', CHAR_nested_field),
            ('raw_document', data.RawField()),
            ('question', WORD_field),
            ('question_char', CHAR_nested_field),
            ('raw_question', data.RawField()),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('ext_answer', data.RawField()),
            ('gt_answer', data.RawField())
        ]
