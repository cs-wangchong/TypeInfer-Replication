import logging
import json

from typeinfer.model import Model
from typeinfer.utils import init_log

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "models"

    init_log(level=logging.INFO)
    
    model = Model(
        generator_model_name="Salesforce/codet5-base",
        ranker_model_name=None,
        generator_ckpt=None,
        ranker_ckpt=None,
        device="cuda"
    )
    
    
    masked_codes_dict = json.load(open(f"{DATA_DIR}/trainset_masked_source_codet5.json", "r"))
    answers_dict = json.load(open(f"{DATA_DIR}/trainset_transformed.json", "r"))
    user_types_dict = json.load(open(f"{DATA_DIR}/trainset_usertypes.json", "r"))

    pairs = list(masked_codes_dict.items())
    pairs = [(_id, code) for _id, code in pairs if len(set(user_types_dict[_id][1]) - {answers_dict[_id][1]}) > 0]

    ids, masked_codes = zip(*pairs)
    gt_types = [answers_dict[_id][1] for _id in ids]
    user_types_list = [user_types_dict[_id][1] for _id in ids]
    
    model.train_generator(
        masked_codes,
        gt_types,
        epochs=3,
        train_batch_size=4,
        valid_batch_size=8,
        valid_ratio=0.2,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=100,
        valid_step=0,
        save_dir=f"{OUTPUT_DIR}",
        best_k=3,
      )

    ranking_examples = model.build_ranker_data(masked_codes=masked_codes, expected_types=gt_types, user_types_list=user_types_list)
    
    model = Model(
        generator_model_name=None,
        ranker_model_name="Salesforce/codet5-base",
        generator_ckpt=None,
        ranker_ckpt=None,
        device="cuda"
    )

    model.train_ranker(
        ranking_examples,
        epochs=3,
        train_batch_size=4,
        valid_batch_size=8,
        valid_ratio=0.2,
        learning_rate=1e-5,
        margin=0.5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=100,
        valid_step=0,
        save_dir=f"{OUTPUT_DIR}",
        best_k=3,
      )