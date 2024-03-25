import json
import logging
import random
import math
import os
from pathlib import Path
from typing import Union
import numpy as np
import typing

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

from typeinfer.utils import Bleu
from typeinfer.info_nce import InfoNCE

MASK = "<extra_id_0>"

BUILTINS = set(typing.__all__)
BUILTINS.update(t.lower() for t in BUILTINS.copy())
# The following builtins are from ManyTypes4Py
BUILTINS.update({'IO', 'Literal', 'Warning', 'DefaultDict', 'ChainMap', 'bytearray', 'Pattern', 'Optional', 'Exception', 'str', 'tuple', 'Deque', 'Collection', 'Iterable', 'bool', 'Match', 'complex', 'dict', 'Any', 'Type', 'list', 'Counter', 'frozenset', 'None', 'Reversible', 'Iterator', 'Union', 'set', 'int', 'bytes', 'Generator', 'Callable', 'memoryview', 'float'})


class GenerationModel:
    def __init__(
        self,
        model_name="Salesforce/codet5-base",
        ckpt_path=None,
        device="cuda"
    ):
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.device = device
    
    def train(
        self,
        masked_codes,
        expected_types,
        epochs=5,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=5000,
        save_dir=None,
        best_k=3,

    ):
        all_examples = list(zip(masked_codes, expected_types))
        random.shuffle(all_examples)
        sep_idx = int(len(all_examples) * (1 - valid_ratio))
        train_examples, valid_examples = all_examples[:sep_idx], all_examples[sep_idx:]
        num_training_batchs = math.ceil(len(train_examples) / train_batch_size)
        num_training_steps = epochs * num_training_batchs

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f"train examples: {len(train_examples)}")
        logging.info(f"valid examples: {len(valid_examples)}")
        logging.info(f"epochs: {epochs}")
        logging.info(f"learning rate: {learning_rate}")
        logging.info(f"train batch size: {train_batch_size}")
        logging.info(f"valid batch size: {valid_batch_size}")
        logging.info(f"train batch num: {num_training_batchs}")
        logging.info(f"valid step num: {num_training_steps}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")

        logging.info("")
        logging.info("")

        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model_parameters, lr=learning_rate, eps=adam_epsilon)
        if warmup < 1:
            num_warmup_steps = num_training_steps * warmup
        else:
            num_warmup_steps = int(warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        total_steps = 0
        best_ckpts = []
        valid_step = num_training_batchs if valid_step == 0 else valid_step
        for cur_epoch in range(epochs):
            self.model.train()
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(train_examples), train_batch_size), range(train_batch_size, len(train_examples)+train_batch_size, train_batch_size)))
            batch_ranges = tqdm(batch_ranges, desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = train_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets = zip(*batch)
                targets = [f"{MASK}{target}" for target in targets]

                source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                target_ids = target_ids.to(self.device)

#                 target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                # y_ids = target_ids[:, :-1].contiguous()
                # labels = target_ids[:, 1:].clone().detach()
                # labels[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=attention_mask,
                    # decoder_input_ids=y_ids,
                    # labels=labels,
                    labels=target_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )

                loss = outputs.loss   
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_training_batchs == 0 or total_steps == num_training_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epochs}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_training_steps:
                    bleu = self.evaluate(valid_examples)
                    self.model.train()
                    logging.info(f"[Validation] Step {total_steps}: bleu-4 {round(bleu, 4)}")

                    if save_dir is None:
                        continue
                    
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, bleu))
                    elif bleu > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, bleu)

                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"Best checkpoints: {best_ckpts}")
                    # best_ckpt = best_ckpts[0][0]
                    json.dump(best_ckpts, open(f"{save_dir}/model-ckpts.json", "w"), indent=4)
        del self.model

    def evaluate(self, eval_examples, valid_batch_size=16, gen_max_len=10):
        self.model.eval()
        predictions, expectations = [], []
        batch_ranges = list(zip(range(0, len(eval_examples), valid_batch_size), range(valid_batch_size, len(eval_examples)+valid_batch_size, valid_batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Validation"):
                batch = eval_examples[beg:end]
                sources, targets = zip(*batch)
                expectations.extend(targets)

                targets = [f"{MASK}{target}" for target in targets]
                
                source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.model.generate(
                    source_ids,
                    attention_mask=attention_mask,
                    max_length=gen_max_len,
                    repetition_penalty=2.5,
                    num_beams=2
                )
                outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
                predictions.extend(outputs)

        predictions = [self.tokenizer.tokenize(item.strip()) for item in predictions]
        expectations = [self.tokenizer.tokenize(item.strip()) for item in expectations]

        bleu, *_ = Bleu.compute_bleu(expectations, predictions, smooth=True)
        return bleu
    

    def predict(self, masked_codes, k=5, max_len=25, batch_size=32):
        self.model.eval()
        data_size = len(masked_codes)
        predictions = []
        with torch.no_grad():
            for beg, end in tqdm(list(zip(range(0, data_size, batch_size), range(batch_size, data_size + batch_size, batch_size))), desc="generating", ascii=True):
                batch_masked_codes = masked_codes[beg:end]
                input_ids = self.tokenizer(batch_masked_codes, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                input_ids = input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_length=max_len, num_beams=k, num_return_sequences=k, return_dict_in_generate=True, output_scores=True)
                
                generations = self.tokenizer.batch_decode(outputs.sequences)
                if hasattr(outputs, "sequences_scores"):
                    scores = [float(s) for s in outputs.sequences_scores.cpu().numpy().tolist()]
                else:
                    scores = torch.cat([step.unsqueeze(dim=1) for step in outputs.scores], dim=1)
                    scores = F.log_softmax(scores, -1)
                    mask = outputs.sequences[:, 1:].ne(self.tokenizer.pad_token_id)
                    scores = torch.gather(scores, dim=2, index=outputs.sequences[:, 1:].unsqueeze(-1))
                    scores = scores.squeeze(-1)
                    scores = scores * mask
                    scores = scores.sum(dim=-1)
                    scores = [float(l) for l in scores.cpu().numpy().tolist()] 
                
                batch_gens_list = [list(zip(generations[b:e], scores[b:e])) for b, e in zip(range(0, len(generations), k), range(k, len(generations) + k, k)) ]

                for masked_code, gens in zip(batch_masked_codes, batch_gens_list):
                    logging.debug("----- masked code -----")
                    logging.debug(masked_code)

                    cleaned_gens = list()
                    for gen, score in gens:
                        gen = gen.split("<extra_id_0>")[-1].split("<extra_id_1>")[0]
                        gen = self.tokenizer.decode(self.tokenizer.encode(gen), skip_special_tokens=True)
                        if gen.count("[") != gen.count("]"):
                            gen = gen.split("[")[0]
                        score = np.exp(score)
                        cleaned_gens.append((gen, score))
                    predictions.append(cleaned_gens)
        return predictions
    
    def compute_likelihood(self, masked_codes, cands_list, batch_size=48):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for masked_code, cands in tqdm(zip(masked_codes, cands_list), total=len(masked_codes), desc="Likelihood", ascii=True):
                likelihoods = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = [f"{MASK}{cand}" for cand in cands[beg:end]]
                    source_ids = self.tokenizer([masked_code] * len(batch_cands), add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    target_ids = self.tokenizer(batch_cands, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.device)

                    attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                    decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                    
                    outputs = self.model(input_ids=source_ids, attention_mask=attention_mask, labels=target_ids, decoder_attention_mask=decoder_attention_mask, output_hidden_states=True)
                    
                    # logits = F.softmax(outputs.logits, -1)
                    # target_logits = torch.gather(logits, dim=2, index=target_ids.unsqueeze(-1))
                    # target_logits = target_logits.squeeze(-1)
                    # target_logits = target_logits * decoder_attention_mask
                    # likelihoods = target_logits.sum(dim=-1) / decoder_attention_mask.sum(dim=-1)
                    # likelihoods = [float(l) for l in likelihoods.cpu().numpy().tolist()]
                    
                    logits = F.log_softmax(outputs.logits, -1)
                    target_logits = torch.gather(logits, dim=2, index=target_ids.unsqueeze(-1))
                    target_logits = target_logits.squeeze(-1)
                    target_logits = target_logits * decoder_attention_mask
                    _likelihoods = target_logits.sum(dim=-1)
                    _likelihoods = torch.exp(_likelihoods) 
                    _likelihoods = [float(l) for l in _likelihoods.cpu().numpy().tolist()] 
                    likelihoods.extend(_likelihoods)
                predictions.append(likelihoods)
        return predictions



class SimilarityModel:
    def __init__(
        self,
        model_name="Salesforce/codet5-base",
        ckpt_path=None,
        # infer_ckpt=None,
        embedding_mode="msk",
        device="cuda"
    ):
        assert embedding_mode in {"avg", "msk"}, "only support `avg` and `msk` embeding mode"
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.mask_id = self.tokenizer._convert_token_to_id(MASK)
        self.embedding_mode = embedding_mode

        self.device = device
    
    def build_training_data(
        self,
        masked_codes,
        expected_types,
        user_types_list,
        generations_list,
        neg_k=4,
        balanced=False
    ):
        if balanced:
            u_examples, nu_examples = [], []
            for masked_code, expected_type, user_types, gens in zip(masked_codes, expected_types, user_types_list, generations_list):
                if expected_type in set(user_types):
                    u_examples.append((masked_code, expected_type, user_types, gens))
                else:
                    nu_examples.append((masked_code, expected_type, user_types, gens))
            count = min(len(u_examples), len(nu_examples))
            examples = random.sample(u_examples, count) + random.sample(nu_examples, count)
            random.shuffle(examples)
            masked_codes, expected_types, user_types_list, generations_list = zip(*examples)

        all_examples = []
        for masked_code, expected_type, user_types, gens in tqdm(zip(masked_codes, expected_types, user_types_list, generations_list), total=len(masked_codes), desc="Building CLR Data", ascii=True):
            dedup_gens = []
            for gen, _ in gens:
                if gen == expected_type or gen in dedup_gens:
                    continue
                dedup_gens.append(gen)
            gens = dedup_gens
            userdefs = set() if user_types is None else set(user_types)
            userdefs = [userdef for userdef in userdefs if userdef != expected_type]

            if len(gens) == 0 and len(userdefs) == 0:
                negs = ['TYPE'] * neg_k
                all_examples.append((masked_code, expected_type, negs))
                continue

            negs = list()
            '''strategy-1: random'''
            if len(gens) > 0:
                neg = random.choice(gens)
                negs.append(neg)
                gens.remove(neg)
            rest_num = neg_k - len(negs)
            cands = gens + userdefs
            negs.extend(random.sample(cands, min(rest_num, len(cands))))
            if len(negs) < neg_k:
                negs = negs * math.ceil(neg_k / len(negs))
                negs = random.sample(negs, neg_k)
            all_examples.append((masked_code, expected_type, negs))
            
            '''strategy-2: bleu similarity'''
            # negs.extend(gens[:neg_k // 2])
            # negs.extend(random.sample(userdefs, min(neg_k - neg_k // 2, len(userdefs))))
            # if len(negs) < neg_k:
            #     negs = negs * math.ceil(neg_k / len(negs))
            #     negs = random.sample(negs, neg_k)
            # all_examples.append((masked_code, expected_type, negs))

            '''strategy-3: bleu similarity'''
            # tgt_tokens = self.tokenizer.tokenize(tgt)
            # bleus = [Bleu.compute_bleu([tgt_tokens], [self.tokenizer.tokenize(neg)], max_order=2)[0] for neg in negs]
            # hard_neg, max_bleu = max(zip(negs, bleus), key=lambda p: p[-1])
            # if max_bleu < bleu_thres:
            #     continue
            # all_examples.append((masked_code, expected_type, hard_neg))
        return all_examples
    

    
    def train(
        self,
        masked_codes,
        expected_types,
        user_types_list,
        generations_list,
        neg_k=4,
        balanced=False,
        refresh_clr_data=False,
        epochs=5,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        margin=0.2,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=5000,
        save_dir=None,
        best_k=3,

    ):  
        all_examples = list(zip(masked_codes, expected_types, user_types_list, generations_list))
        random.shuffle(all_examples)
        sep_idx = int(len(all_examples) * (1 - valid_ratio))
        train_examples, valid_examples = all_examples[:sep_idx], all_examples[sep_idx:]
        
        num_training_batchs = math.ceil(len(train_examples) / train_batch_size)
        num_valid_batchs = math.ceil(len(valid_examples) / valid_batch_size)
        num_training_steps = epochs * num_training_batchs

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f"epochs: {epochs}")
        logging.info(f"learning rate: {learning_rate}")
        logging.info(f"train batch size: {train_batch_size}")
        logging.info(f"valid batch size: {valid_batch_size}")
        # logging.info(f"generation k: {gen_k}")
        # logging.info(f"generation max length: {gen_max_len}")
        # logging.info(f"generation batch size: {gen_batch_size}")
        # logging.info(f"bleu threshold: {bleu_thres}")
        logging.info(f"margin: {margin}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")
        logging.info("")
        logging.info("")
        logging.info(f"train examples: {len(train_examples)}")
        logging.info(f"valid examples: {len(valid_examples)}")
        logging.info(f"train batch num: {num_training_batchs}")
        logging.info(f"valid batch num: {num_valid_batchs}")
        logging.info("")
        logging.info("")

        # frozen_params = set()
        # for name, param in self.model.encoder.named_parameters():
        #     param.requires_grad = False
        #     frozen_params.add(name)
        #     print(f"param: {name} is frozen...")
        # for block in self.model.decoder.block[:-1]:
        #     for name, param in block.named_parameters():
        #         param.requires_grad = False
        #         frozen_params.add(name)
        #         print(f"param: {name} is frozen...")
        # active_params = [(n, p) for n, p in self.model.named_parameters() if n not in frozen_params]
        active_params = [(n, p) for n, p in self.model.named_parameters()]
        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in active_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in active_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup < 1:
            num_warmup_steps = num_training_steps * warmup
        else:
            num_warmup_steps = int(warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        info_nce = InfoNCE(negative_mode='paired')

        if not refresh_clr_data:
            clr_examples = self.build_training_data(
                *zip(*train_examples),
                neg_k=neg_k,
                balanced=balanced,
            )

        total_steps = 0
        best_ckpts = []
        valid_step = num_training_batchs if valid_step == 0 else valid_step
        for cur_epoch in range(epochs):
            if refresh_clr_data:
                clr_examples = self.build_training_data(
                    *zip(*train_examples),
                    neg_k=neg_k,
                    balanced=balanced,
                )
            self.model.train()
            random.shuffle(clr_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(clr_examples), train_batch_size), range(train_batch_size, len(clr_examples)+train_batch_size, train_batch_size)))
            batch_ranges = tqdm(batch_ranges, desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = clr_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets, negatives_list = zip(*batch)
                targets = [f"{MASK}{target}" for target in targets]
                negatives_list = [[f"{MASK}{neg}" for neg in negs] for negs in negatives_list]

                source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                target_ids = target_ids.to(self.device)
                
                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=attention_mask,
                    # decoder_input_ids=y_ids,
                    # labels=labels,
                    labels=target_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
                logging.debug(outputs.encoder_last_hidden_state.shape)
                hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                logging.debug(encoder_features.shape)
                if self.embedding_mode == 'avg':
                    length = torch.sum(attention_mask, axis=1) - 1
                    encoder_features = torch.sum(encoder_features, axis=1)
                    encoder_features = encoder_features * (1 / length.unsqueeze(-1))
                else:
                    mask_pos = source_ids.eq(self.mask_id).long().nonzero(as_tuple=True)
                    mask_pos = (mask_pos[0], mask_pos[1] + 1)
                    encoder_features = encoder_features[mask_pos]

                target_decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                logging.debug(target_decoder_features.shape)
                if self.embedding_mode == 'avg':
                    length = torch.sum(decoder_attention_mask, axis=1) - 1
                    target_decoder_features = torch.sum(target_decoder_features, axis=1)
                    target_decoder_features = target_decoder_features * (1 / length.unsqueeze(-1))
                else:
                    eos_pos = target_ids.eq(self.tokenizer.eos_token_id).long().nonzero(as_tuple=True)
                    target_decoder_features = target_decoder_features[eos_pos]
                
                negative_decoder_features = []
                for source, negatives in zip(sources, negatives_list):
                    _sources = [source] * neg_k
                    source_ids = self.tokenizer(_sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    negative_ids = self.tokenizer(negatives, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    negative_ids = negative_ids.to(self.device)

                    attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                    decoder_attention_mask = negative_ids.ne(self.tokenizer.pad_token_id)
                    outputs = self.model(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        # decoder_input_ids=y_ids,
                        # labels=labels,
                        labels=negative_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    logging.debug(decoder_features.shape)
                    if self.embedding_mode == 'avg':
                        length = torch.sum(decoder_attention_mask, axis=1) - 1
                        decoder_features = torch.sum(decoder_features, axis=1)
                        decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    else:
                        eos_pos = negative_ids.eq(self.tokenizer.eos_token_id).long().nonzero(as_tuple=True)
                        decoder_features = decoder_features[eos_pos]
                    negative_decoder_features.append(decoder_features)
                negative_decoder_features = torch.stack(negative_decoder_features, dim=0)

                if (encoder_features.shape[0] == target_decoder_features.shape[0] == negative_decoder_features.shape[0]):
                    loss = info_nce(encoder_features, target_decoder_features, negative_decoder_features)

                    # loss = F.margin_ranking_loss(
                    #     target_cos,
                    #     negative_cos,
                    #     torch.ones(target_cos.size(),
                    #     device=target_cos.device),
                    #     margin=margin,
                    #     reduction="mean"
                    # )
        
                    train_loss += loss.item()

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_training_batchs == 0 or total_steps == num_training_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epochs}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_training_steps:
                    top1_acc = self.evaluate(valid_examples)
                    self.model.train()
                    logging.info(f"[Validation] Step {total_steps}: top1 acc {round(top1_acc, 4)}")

                    if save_dir is None:
                        continue
                    
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, top1_acc))
                    elif top1_acc > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, top1_acc)

                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"Best checkpoints: {best_ckpts}")
                    # best_ckpt = best_ckpts[0][0]
                    json.dump(best_ckpts, open(f"{save_dir}/model-ckpts.json", "w"), indent=4)
        del self.model
    
    def evaluate(self, eval_examples, batch_size=48):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for source, target, userdefs, gens in tqdm(eval_examples, ascii=True, desc="Validation"):
                userdefs = [] if userdefs is None else list(userdefs)
                cands = list({gen for gen, _ in gens} | set(userdefs) | {target})

                rankings = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = cands[beg:end]
                    sources = [source] * len(batch_cands)
                    targets = [f"{MASK}{cand}" for cand in batch_cands]
                    source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.device)
                    # target_ids[target_ids == self.tokenizer.pad_token_id] = -100
                    attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                    decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                    outputs = self.model(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        # decoder_input_ids=y_ids,
                        # labels=labels,
                        labels=target_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                    hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                    encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    if self.embedding_mode == 'avg':
                        length = torch.sum(attention_mask, axis=1) - 1
                        encoder_features = torch.sum(encoder_features, axis=1)
                        encoder_features = encoder_features * (1 / length.unsqueeze(-1))
                    else:
                        mask_pos = torch.tensor(range(source_ids.shape[0])), torch.argmax(source_ids.eq(self.mask_id).long(), -1) + 1
                        encoder_features = encoder_features[mask_pos]

                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    if self.embedding_mode == 'avg':
                        length = torch.sum(decoder_attention_mask, axis=1) - 1
                        decoder_features = torch.sum(decoder_features, axis=1)
                        decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    else:
                        eos_pos = torch.tensor(range(target_ids.shape[0])), torch.argmax(target_ids.eq(self.tokenizer.eos_token_id).long(), -1)
                        decoder_features = decoder_features[eos_pos]
                    
                    target_cos = torch.cosine_similarity(encoder_features, decoder_features)
                    cand_scores = [float(l) for l in target_cos.cpu().numpy().tolist()]
                    rankings.extend([(cand, l) for cand, l in zip(batch_cands, cand_scores)])
                rankings.sort(key=lambda x:x[-1], reverse=True)
                r = [cand for cand, _ in rankings].index(target) + 1
                predictions.append(r)
        top1_acc = len([r for r in predictions if r == 1]) / len(predictions)
        return top1_acc
    
    
    def compute_similarity(self, masked_codes, cands_list, batch_size=48):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for masked_code, cands in tqdm(zip(masked_codes, cands_list), total=len(masked_codes), desc="Similarity", ascii=True):
                similarities = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = cands[beg:end]
                    sources = [masked_code] * len(batch_cands)
                    targets = [f"{MASK}{cand}" for cand in batch_cands]
                    source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.device)
                    # target_ids[target_ids == self.tokenizer.pad_token_id] = -100
                    attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                    decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                    outputs = self.model(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        # decoder_input_ids=y_ids,
                        # labels=labels,
                        labels=target_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                    hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                    encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    if self.embedding_mode == 'avg':
                        length = torch.sum(attention_mask, axis=1) - 1
                        encoder_features = torch.sum(encoder_features, axis=1)
                        encoder_features = encoder_features * (1 / length.unsqueeze(-1))
                    else:
                        mask_pos = torch.tensor(range(source_ids.shape[0])), torch.argmax(source_ids.eq(self.mask_id).long(), -1) + 1
                        encoder_features = encoder_features[mask_pos]

                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    if self.embedding_mode == 'avg':
                        length = torch.sum(decoder_attention_mask, axis=1) - 1
                        decoder_features = torch.sum(decoder_features, axis=1)
                        decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    else:
                        eos_pos = torch.tensor(range(target_ids.shape[0])), torch.argmax(target_ids.eq(self.tokenizer.eos_token_id).long(), -1)
                        decoder_features = decoder_features[eos_pos]
                    
                    _similarities = torch.cosine_similarity(encoder_features, decoder_features)
                    _similarities = [float(sim) for sim in _similarities.cpu().numpy().tolist()]
                    similarities.extend(_similarities)
                predictions.append(similarities)
        return predictions
    

class BiEncoderSimilarityModel:
    def __init__(
        self,
        model_name="Salesforce/codet5-base",
        ckpt_path=None,
        embedding_mode='avg',
        device="cuda"
    ):
        
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.code_encoder: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        self.type_encoder: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        if ckpt_path:
            self.type_encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.code_encoder.to(device)
        self.type_encoder.to(device)
        self.embedding_mode = embedding_mode
        self.device = device
        self.mask_id = self.tokenizer._convert_token_to_id(MASK)
    
    def build_training_data(
        self,
        masked_codes,
        expected_types,
        user_types_list,
        generations_list,
        neg_k=4,
        balanced=False
    ):
        if balanced:
            u_examples, nu_examples = [], []
            for masked_code, expected_type, user_types, gens in zip(masked_codes, expected_types, user_types_list, generations_list):
                if expected_type in set(user_types):
                    u_examples.append((masked_code, expected_type, user_types, gens))
                else:
                    nu_examples.append((masked_code, expected_type, user_types, gens))
            count = min(len(u_examples), len(nu_examples))
            examples = random.sample(u_examples, count) + random.sample(nu_examples, count)
            random.shuffle(examples)
            masked_codes, expected_types, user_types_list, generations_list = zip(*examples)

        all_examples = []
        for masked_code, expected_type, user_types, gens in tqdm(zip(masked_codes, expected_types, user_types_list, generations_list), total=len(masked_codes), desc="Building CLR Data", ascii=True):
            dedup_gens = []
            for gen, _ in gens:
                if gen == expected_type or gen in dedup_gens:
                    continue
                dedup_gens.append(gen)
            gens = dedup_gens
            userdefs = set() if user_types is None else set(user_types)
            userdefs = [userdef for userdef in userdefs if userdef != expected_type]

            if len(gens) == 0 and len(userdefs) == 0:
                negs = ['TYPE'] * neg_k
                all_examples.append((masked_code, expected_type, negs))
                continue

            negs = list()
            '''strategy-1: random'''
            if len(gens) > 0:
                neg = random.choice(gens)
                negs.append(neg)
                gens.remove(neg)
            rest_num = neg_k - len(negs)
            cands = gens + userdefs
            negs.extend(random.sample(cands, min(rest_num, len(cands))))
            if len(negs) < neg_k:
                negs = negs * math.ceil(neg_k / len(negs))
                negs = random.sample(negs, neg_k)
            all_examples.append((masked_code, expected_type, negs))
            
            '''strategy-2: bleu similarity'''
            # negs.extend(gens[:neg_k // 2])
            # negs.extend(random.sample(userdefs, min(neg_k - neg_k // 2, len(userdefs))))
            # if len(negs) < neg_k:
            #     negs = negs * math.ceil(neg_k / len(negs))
            #     negs = random.sample(negs, neg_k)
            # all_examples.append((masked_code, expected_type, negs))

            '''strategy-3: bleu similarity'''
            # tgt_tokens = self.tokenizer.tokenize(tgt)
            # bleus = [Bleu.compute_bleu([tgt_tokens], [self.tokenizer.tokenize(neg)], max_order=2)[0] for neg in negs]
            # hard_neg, max_bleu = max(zip(negs, bleus), key=lambda p: p[-1])
            # if max_bleu < bleu_thres:
            #     continue
            # all_examples.append((masked_code, expected_type, hard_neg))
        return all_examples
    
    def encode_types(self, types):
        source_ids = self.tokenizer(types, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.device)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.type_encoder(input_ids=source_ids, attention_mask=attention_mask,
                                    labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        if self.embedding_mode == 'avg':
            length = torch.sum(attention_mask, axis=1) - 1
            feats = torch.sum(hidden_states, axis=1)
            feats = feats * (1 / length.unsqueeze(-1))
        else:
            eos_mask = torch.tensor(range(source_ids.shape[0])), torch.argmax(source_ids.eq(self.tokenizer.eos_token_id).long(), -1)
            # if len(torch.unique(eos_mask.sum(1))) > 1:
            #     raise ValueError("All examples must have the same number of <eos> tokens.")
            feats = hidden_states[eos_mask]                                      
        return feats

    def encode_codes(self, masked_codes):
        source_ids = self.tokenizer(masked_codes, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.device)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = self.code_encoder(input_ids=source_ids, attention_mask=attention_mask,
                                        labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        if self.embedding_mode == 'avg':
            length = torch.sum(attention_mask, axis=1) - 1
            feats = torch.sum(hidden_states, axis=1)
            feats = feats * (1 / length.unsqueeze(-1))
        else:
            mask_pos = torch.tensor(range(source_ids.shape[0])), torch.argmax(source_ids.eq(self.mask_id).long(), -1) + 1
            feats = hidden_states[mask_pos]
        return feats
    
    def train(
        self,
        masked_codes,
        expected_types,
        user_types_list,
        generations_list,
        neg_k=4,
        balanced=False,
        refresh_clr_data=False,
        epochs=5,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        margin=0.2,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=5000,
        save_dir=None,
        best_k=3,

    ):  
        all_examples = list(zip(masked_codes, expected_types, user_types_list, generations_list))
        random.shuffle(all_examples)
        sep_idx = int(len(all_examples) * (1 - valid_ratio))
        train_examples, valid_examples = all_examples[:sep_idx], all_examples[sep_idx:]
        
        num_training_batchs = math.ceil(len(train_examples) / train_batch_size)
        num_valid_batchs = math.ceil(len(valid_examples) / valid_batch_size)
        num_training_steps = epochs * num_training_batchs

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f"epochs: {epochs}")
        logging.info(f"learning rate: {learning_rate}")
        logging.info(f"train batch size: {train_batch_size}")
        logging.info(f"valid batch size: {valid_batch_size}")
        # logging.info(f"generation k: {gen_k}")
        # logging.info(f"generation max length: {gen_max_len}")
        # logging.info(f"generation batch size: {gen_batch_size}")
        # logging.info(f"bleu threshold: {bleu_thres}")
        logging.info(f"margin: {margin}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")
        logging.info("")
        logging.info("")
        logging.info(f"train examples: {len(train_examples)}")
        logging.info(f"valid examples: {len(valid_examples)}")
        logging.info(f"train batch num: {num_training_batchs}")
        logging.info(f"valid batch num: {num_valid_batchs}")
        logging.info("")
        logging.info("")

        # frozen_params = set()
        # for name, param in self.model.encoder.named_parameters():
        #     param.requires_grad = False
        #     frozen_params.add(name)
        #     print(f"param: {name} is frozen...")
        # for block in self.model.decoder.block[:-1]:
        #     for name, param in block.named_parameters():
        #         param.requires_grad = False
        #         frozen_params.add(name)
        #         print(f"param: {name} is frozen...")
        # active_params = [(n, p) for n, p in self.model.named_parameters() if n not in frozen_params]
        for param in self.code_encoder.parameters():
            param.requires_grad = False
        self.code_encoder.eval()
        active_params = [(n, p) for n, p in self.type_encoder.named_parameters()]
        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in active_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in active_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup < 1:
            num_warmup_steps = num_training_steps * warmup
        else:
            num_warmup_steps = int(warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        info_nce = InfoNCE(negative_mode='paired')

        if not refresh_clr_data:
            clr_examples = self.build_training_data(
                *zip(*train_examples),
                neg_k=neg_k,
                balanced=balanced,
            )

        total_steps = 0
        best_ckpts = []
        valid_step = num_training_batchs if valid_step == 0 else valid_step
        for cur_epoch in range(epochs):
            if refresh_clr_data:
                clr_examples = self.build_training_data(
                    *zip(*train_examples),
                    neg_k=neg_k,
                    balanced=balanced,
                )
            self.type_encoder.train()
            random.shuffle(clr_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(clr_examples), train_batch_size), range(train_batch_size, len(clr_examples)+train_batch_size, train_batch_size)))
            batch_ranges = tqdm(batch_ranges, desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = clr_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets, negatives_list = zip(*batch)
                # targets = [f"{MASK}{target}" for target in targets]
                # negatives_list = [[f"{MASK}{neg}" for neg in negs] for negs in negatives_list]

                with torch.no_grad():
                    code_feats = self.encode_codes(sources)

                target_feats = self.encode_types(targets)
                
                negative_features = []
                for negatives in negatives_list:
                    negative_features.append(self.encode_types(negatives))
                negative_features = torch.stack(negative_features, dim=0)

                if (code_feats.shape[0] == target_feats.shape[0] == negative_features.shape[0]):
                    loss = info_nce(code_feats, target_feats, negative_features)
                    train_loss += loss.item()

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.type_encoder.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_training_batchs == 0 or total_steps == num_training_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epochs}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_training_steps:
                    top1_acc = self.evaluate(valid_examples)
                    self.type_encoder.train()
                    logging.info(f"[Validation] Step {total_steps}: top1 acc {round(top1_acc, 4)}")

                    if save_dir is None:
                        continue
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.type_encoder.module if hasattr(self.type_encoder, 'module') else self.type_encoder
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, top1_acc))
                    elif top1_acc > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.type_encoder.module if hasattr(self.type_encoder, 'module') else self.type_encoder
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        torch.save(model_to_save.state_dict(), f"{save_dir}/model-best.ckpt")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, top1_acc)

                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"Best checkpoints: {best_ckpts}")
                    # best_ckpt = best_ckpts[0][0]
                    json.dump(best_ckpts, open(f"{save_dir}/model-ckpts.json", "w"), indent=4)
        del self.type_encoder
    
    def evaluate(self, eval_examples, batch_size=48):
        self.type_encoder.eval()
        predictions = []
        with torch.no_grad():
            for source, target, userdefs, gens in tqdm(eval_examples, ascii=True, desc="Validation"):
                userdefs = [] if userdefs is None else list(userdefs)
                cands = list({gen for gen, _ in gens} | set(userdefs) | {target})

                rankings = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = cands[beg:end]
                    sources = [source] * len(batch_cands)
                    code_feats = self.encode_codes(sources)
                    cand_feats = self.encode_types(batch_cands)
                    target_cos = torch.cosine_similarity(code_feats, cand_feats)
                    cand_scores = [float(l) for l in target_cos.cpu().numpy().tolist()]
                    rankings.extend([(cand, l) for cand, l in zip(batch_cands, cand_scores)])
                rankings.sort(key=lambda x:x[-1], reverse=True)
                r = [cand for cand, _ in rankings].index(target) + 1
                predictions.append(r)
        top1_acc = len([r for r in predictions if r == 1]) / len(predictions)
        return top1_acc
    
    
    def compute_similarity(self, masked_codes, cands_list, batch_size=48):
        self.type_encoder.eval()
        predictions = []
        with torch.no_grad():
            for masked_code, cands in tqdm(zip(masked_codes, cands_list), total=len(masked_codes), desc="Similarity", ascii=True):
                similarities = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = cands[beg:end]
                    sources = [masked_code] * len(batch_cands)
                    code_feats = self.encode_codes(sources)
                    cand_feats = self.encode_types(batch_cands)
                    _similarities = torch.cosine_similarity(code_feats, cand_feats)
                    _similarities = [float(l) for l in _similarities.cpu().numpy().tolist()]
                    similarities.extend(_similarities)
                predictions.append(similarities)
        return predictions


class InferenceModel:
    def __init__(
        self,
        generation_model: GenerationModel,
        similarity_model: Union[SimilarityModel,BiEncoderSimilarityModel]
    ):
        self.generation_model = generation_model
        self.similarity_model = similarity_model


    def infer(self, masked_codes, user_types_list, generating_list=None, alpha=0.5, gen_k=5, gen_max_len=10, gen_batch_size=12, rank_batch_size=64):
        if generating_list is None:
            generating_list = self.generation_model.predict(masked_codes, gen_k, gen_max_len, gen_batch_size)

        user_types_list = [(set() if userdefs is None else set(userdefs)) for userdefs in user_types_list]

        cands_list = []
        for gens, userdefs in zip(generating_list, user_types_list):
            cleaned_gens = []
            bases = set()
            for gen, _ in gens:
                base = gen.split("[")[0].split(".")[-1]
                if base not in BUILTINS and len(userdefs) > 0 and base not in userdefs:
                    continue       
                # # only keep one base type to benefit user-defined types
                # if base in bases:
                #     continue
                # bases.add(base)
                cleaned_gens.append(gen)
            # # remove the last one to benefit user-defined types
            # if gen_k > 1:
            #     cleaned_gens = cleaned_gens[:gen_k-1]
            cands = list(set(cleaned_gens) | userdefs)
            if len(cands) == 0:
                cands = ["TYPE"]
            cands_list.append(cands)

        likelihoods_list = self.generation_model.compute_likelihood(masked_codes, cands_list, rank_batch_size)
        similarities_list = self.similarity_model.compute_similarity(masked_codes, cands_list, rank_batch_size)

        scores_list = [ 
            [(c, alpha * l + (1-alpha) * s) for c, l, s in zip(cands, likelihoods, similarities)]
            for cands, likelihoods, similarities in zip(cands_list, likelihoods_list, similarities_list)
        ]
        ranking_list = [list(sorted(cands, key=lambda x:x[-1], reverse=True)) for cands in scores_list]
        return generating_list, ranking_list, cands_list, likelihoods_list, similarities_list