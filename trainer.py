import csv
import json
import math
import sys
import time
from collections import defaultdict
from typing import Tuple
import logging
import torch
import torchtext
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Adam
from torchtext.data import BucketIterator, Iterator
from torchtext.vocab import GloVe
from dataset import SquadDataset
from ema import EMA
from evaluate_squad import evaluate
from models.QA_baseline import Baseline
from models.QA_bidaf_nocharemb import BidafSimplified
from models.QA_bidaf_vanilla import BidAF
from util import count_parameters, report_parameters, get_timestamp

import socket


class ModelFramework():
    def ___init__(self):
        pass

    def train_epoch(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer,
                    train_iter: Iterator) -> float:
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            logprobs_S, logprobs_E = model(batch)
            loss_s = lossfunction(logprobs_S, batch.a_start)
            loss_e = lossfunction(logprobs_E, batch.a_end)
            loss = loss_s + loss_e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            if i % 300 == 0 and i > 0:
                logging.info(f"Training loss: {train_loss / i + 1}")
        return train_loss / len(train_iter.data())

    def get_spans(self, batch, candidates):
        r = []
        for i in range(len(batch.raw_document_context)):
            candidate_start = candidates[0][i]
            candidates_end = candidates[1][i]
            if candidate_start > len(batch.document_token_positions[i]) - 1:
                candidate_start = len(batch.document_token_positions[i]) - 1
            if candidates_end > len(batch.document_token_positions[i]) - 1:
                candidates_end = len(batch.document_token_positions[i]) - 1

            r.append(batch.raw_document_context[i][batch.document_token_positions[i][candidate_start][0]:
                                                   batch.document_token_positions[i][candidates_end][-1]])
        return r

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, iter: Iterator, ema=None, log_results=False) -> \
            Tuple[float, float, float]:
        model.eval()
        if ema is not None:
            backup_params = EMA.ema_backup_and_loadavg(ema, model)

        results = dict()
        ids = []
        lossvalues = []
        spans = []
        gt_spans = []
        span_probs = []
        for i, batch in enumerate(iter):
            ids += batch.id
            logprobs_S, logprobs_E = model(batch)
            loss_s = lossfunction(logprobs_S, batch.a_start)
            loss_e = lossfunction(logprobs_E, batch.a_end)
            loss = loss_s + loss_e
            lossvalues += loss.tolist()

            best_span_probs, candidates = model.decode(logprobs_S, logprobs_E)
            span_probs += best_span_probs.tolist()
            spans += self.get_spans(batch, candidates)
            gt_spans += batch.gt_answer

        # compute the final loss and results
        # we need to filter trhough multiple possible choices and pick the best one
        lossdict = defaultdict(lambda: math.inf)
        probs = defaultdict(lambda: 0)
        for id, value, span, span_prob in zip(ids, lossvalues, spans, span_probs):
            # record only lowest loss
            if lossdict[id] > value:
                lossdict[id] = value
            results[id] = span
            probs[id] = span_prob

        if log_results:
            self.log_results(results, probs)

        loss = sum(lossdict.values()) / len(lossdict)
        prediction_file = f".data/squad/dev_results_{socket.gethostname()}.json"
        with open(prediction_file, "w") as f:
            json.dump(results, f)

        dataset_file = ".data/squad/dev-v1.1.json"

        expected_version = '1.1'
        with open(dataset_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            if (dataset_json['version'] != expected_version):
                logging.info('Evaluation expects v-' + expected_version +
                             ', but got dataset with v-' + dataset_json['version'],
                             file=sys.stderr)
            dataset = dataset_json['data']
        with open(prediction_file) as prediction_file:
            predictions = json.load(prediction_file)
        result = evaluate(dataset, predictions)
        logging.info(json.dumps(result))

        if ema is not None:
            EMA.ema_restore_backed_params(backup_params, model)

        return loss, result["exact_match"], result["f1"]

    def fit(self, config, device):
        logging.info(json.dumps(config, indent=4, sort_keys=True))

        if config["char_embeddings"]:
            fields = SquadDataset.prepare_fields_char()
        else:
            fields = SquadDataset.prepare_fields()

        train, val = SquadDataset.splits(fields)
        fields = dict(fields)

        fields["question"].build_vocab(train, val, vectors=GloVe(name='6B', dim=config["embedding_size"]))

        if not type(fields["question_char"]) == torchtext.data.field.RawField:
            fields["question_char"].build_vocab(train, val, max_size=config["char_maxsize_vocab"])

        # Make if shuffle
        train_iter = BucketIterator(train, sort_key=lambda x: -(len(x.question) + len(x.document)),
                                    shuffle=True, sort=False, sort_within_batch=True,
                                    batch_size=config["batch_size"], train=True,
                                    repeat=False,
                                    device=device)

        val_iter = BucketIterator(val, sort_key=lambda x: -(len(x.question) + len(x.document)), sort=True,
                                  batch_size=config["batch_size"],
                                  repeat=False,
                                  device=device)
        #
        # model = torch.load(
        #     "saved/65F1_checkpoint_<class 'trainer.ModelFramework'>_L_2.1954014434733815_2019-06-28_10:06_pcknot2.pt").to(
        #     device)
        if config["modelname"] == "baseline":
            model = Baseline(config, fields["question"].vocab).to(device)
        elif config["modelname"] == "bidaf_simplified":
            model = BidafSimplified(config, fields["question"].vocab).to(device)
        elif config["modelname"] == "bidaf":
            model = BidAF(config, fields['question'].vocab, fields["question_char"].vocab).to(device)
        # glorot_param_init(model)
        logging.info(f"Models has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        if config["optimizer"] == "adam":
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["learning_rate"])
        else:
            raise NotImplementedError(f"Option {config['optimizer']} for \"optimizer\" setting is undefined.")

        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_f1 = 0
            best_em = 0
            ema_active = False
            for it in range(config["max_iterations"]):
                logging.info(f"Iteration {it}")
                if "ema" in config and config["ema"]:
                    ema = EMA.ema_register(config, model)
                    ema_active = True

                self.train_epoch(model, CrossEntropyLoss(), optimizer, train_iter)

                if ema_active:
                    EMA.ema_update(ema, model)

                validation_loss, em, f1 = self.validate(model, CrossEntropyLoss(reduction='none'), val_iter,
                                                        ema=ema if "ema" in config and config[
                                                            "ema"] and ema_active else None)
                if validation_loss < best_val_loss: best_val_loss = validation_loss
                if f1 > best_val_f1: best_val_f1 = validation_loss
                if em > best_em: best_em = em
                logging.info(f"BEST L/F1/EM = {best_val_loss:.2f}/{best_val_f1:.2f}/{best_em:.2f}")
                if em > 65:
                    # Do all this on CPU, this is memory exhaustive!
                    model.to(torch.device("cpu"))

                    if ema_active:
                        # backup current params and load ema params
                        backup_params = EMA.ema_backup_and_loadavg(ema, model)

                        torch.save(model,
                                   f"saved/checkpoint"
                                   f"_{str(self.__class__)}"
                                   f"_EM_{em:.2f}_F1_{f1:.2f}_L_{validation_loss:.2f}_{get_timestamp()}"
                                   f"_{socket.gethostname()}.pt")

                        # load back backed up params
                        EMA.ema_restore_backed_params(backup_params, model)

                    else:
                        torch.save(model,
                                   f"saved/checkpoint"
                                   f"_{str(self.__class__)}"
                                   f"_EM_{em:.2}_F1_{f1:.2}_L_{validation_loss:.2}_{get_timestamp()}"
                                   f"_{socket.gethostname()}.pt")

                    model.to(device)
                logging.info(f"Validation loss: {validation_loss}")

        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def log_results(self, results, probs, val_file=".data/squad/dev-v1.1.json"):
        f = open(f"results/result_{get_timestamp()}_{socket.gethostname()}.csv", mode="w")
        csvw = csv.writer(f, delimiter=',')
        HEADER = ["Correct", "Ground Truth(s)", "Prediction", "Confidence", "Question", "Context", "Topic", "ID"]
        csvw.writerow(HEADER)
        with open(val_file) as fd:
            data_json = json.load(fd)
            for data_topic in data_json["data"]:
                for paragraph in data_topic["paragraphs"]:
                    for question_and_answers in paragraph['qas']:
                        prediction = results[question_and_answers["id"]]
                        confidence = str(f"{probs[question_and_answers['id']]:.2f}")
                        answers = "|".join(map(lambda x: x['text'], question_and_answers['answers']))
                        correct = int(results[question_and_answers["id"]].lower() in map(lambda x: x['text'].lower(),
                                                                                         question_and_answers[
                                                                                             'answers']))
                        ex = [correct,
                              answers,
                              prediction,
                              confidence,
                              question_and_answers['question'],
                              paragraph["context"],
                              data_topic["title"],
                              question_and_answers["id"]]
                        csvw.writerow(ex)
        f.close()
