import logging
import json
from pathlib import Path

from typeinfer.model import GenerationModel, SimilarityModel
from typeinfer.utils import init_log

if __name__ == "__main__":
    RANKING_NEG_K = 4
    RANKING_BALANCE = False
    
    DATA_DIR = "data"
    GENERATION_DIR = "models/generation-model"
    SIMILARITY_DIR = f"models/similarity-model"

    Path(GENERATION_DIR).mkdir(parents=True, exist_ok=True)
    Path(SIMILARITY_DIR).mkdir(parents=True, exist_ok=True)
    
    masked_codes_dict = json.load(open(f"{DATA_DIR}/trainset_masked_source_codet5_truncated.json", "r"))
    answers_dict = json.load(open(f"{DATA_DIR}/trainset_transformed.json", "r"))
    user_types_dict = json.load(open(f"{DATA_DIR}/trainset_usertypes.json", "r"))

    pairs = list(masked_codes_dict.items())
    pairs = [(_id, code) for _id, code in pairs if len(set(user_types_dict[_id][1]) - {answers_dict[_id][1]}) > 0]

    # pairs = random.sample(pairs, 24)

    ids, masked_codes = zip(*pairs)
    gt_types = [answers_dict[_id][1] for _id in ids]
    user_types_list = [user_types_dict[_id][1] for _id in ids]

    init_log(f"{GENERATION_DIR}/training.log", level=logging.INFO)
    gen_model = GenerationModel(
        model_name="Salesforce/codet5-base",
        ckpt_path=None,
        device="cuda"
    )
    
    gen_model.train(
        masked_codes,
        gt_types,
        epochs=3,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=100,
        valid_step=0,
        save_dir=GENERATION_DIR,
        best_k=3,
    )

    gen_model = GenerationModel(
        model_name="Salesforce/codet5-base",
        ckpt_path=f"{GENERATION_DIR}/model-best.ckpt",
        device="cuda"
    )
    gen_predictions = gen_model.predict(
        masked_codes=masked_codes,
        k=5,
        batch_size=48
    )
    gen_predictions = {id: gens for id, gens in zip(ids, gen_predictions)}
    with Path(f"{GENERATION_DIR}/predictions-for-training-samples.json").open("w") as f:
        json.dump(gen_predictions, f, indent=4)

    with Path(f"{GENERATION_DIR}/predictions-for-training-samples.json").open("r") as f:
        gen_predictions = json.load(f)
        gen_predictions = [gen_predictions.get(id, list()) for id in ids]
    
    init_log(f"{SIMILARITY_DIR}/training.log", level=logging.INFO)
    sim_model = SimilarityModel(
        model_name="Salesforce/codet5-base",
        ckpt_path=None,
        embedding_mode="avg",
        device="cuda"
    ) 

    sim_model.train(
        masked_codes,
        gt_types,
        user_types_list,
        gen_predictions,
        neg_k=4,
        balanced=False,
        epochs=3,
        train_batch_size=4,
        valid_batch_size=8,
        valid_ratio=0.1,
        learning_rate=1e-5,
        margin=0.5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=10000,
        save_dir=SIMILARITY_DIR,
        best_k=3,
    )