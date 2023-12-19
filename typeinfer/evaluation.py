import os
import json
from pathlib import Path

from typeinfer.model import Model

if __name__ == "__main__":
    DATA_DIR = "data"
    GENERATOR_PATH = "models/generator-XXX.ckpt"
    RANKER_PATH = "models/ranker-XXX.ckpt"
    OUTPUT_DIR = "output"
    
    model = Model(
        generator_model_name="Salesforce/codet5-base",
        ranker_model_name="Salesforce/codet5-base",
        generator_ckpt=GENERATOR_PATH,
        ranker_ckpt=RANKER_PATH,
        device="cuda"
    )

    masked_codes_dict = json.load(open(f"{DATA_DIR}/testset_masked_source_codet5.json", "r"))
    user_types_dict = json.load(open(f"{DATA_DIR}/testset_usertypes.json", "r"))
    answers_dict = json.load(open(f"{DATA_DIR}/testset_transformed.json", "r"))
    sampled = json.load(open(f"{DATA_DIR}/testset_randomsampled_transformed.json", "r"))
    sampled_ids = {id for id, info in sampled.items()}

    pairs = list(masked_codes_dict.items())
    pairs = list((id, code) for id, code in pairs)
    pairs = [(id, code) for id, code in pairs if id in sampled_ids]
    pairs.sort(key=lambda item:len(item[1]), reverse=True)
    ids, masked_codes = zip(*pairs)
    user_types_list = [user_types_dict[id][1] for id in ids]
    gt_types = [answers_dict[id][1] for id in ids]
    
    print(f"total size: {len(ids)}")
    
    original_predictions, ranking_predictions = model.predict(
        masked_codes,
        user_types_list,
        alpha=0.5,
        gen_k=5,
        gen_max_len=30,
        tem=1.0,
        gen_batch_size=8,
        rank_batch_size=48
    )
    predictions = {id: {"original": orig_preds, "ranking": rank_preds} for id, orig_preds, rank_preds in zip(ids, original_predictions, ranking_predictions)}
    Path(f"{OUTPUT_DIR}/predictions").mkdir(parents=True, exist_ok=True)
    json.dump(predictions, open(f"{OUTPUT_DIR}/predictions/randomsampled.json", "w"), indent=4)