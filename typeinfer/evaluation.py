import json
from pathlib import Path
import time

from typeinfer.model import GenerationModel, SimilarityModel, InferenceModel


if __name__ == "__main__":
    GENERATION_CKPT_PATH = "models/generation-model/model-best.ckpt"
    SIMILARITY_CKPT_PATH = f"models/similarity-model/model-best.ckpt"
    
    DATA_DIR = "data/ManyTypes4Py-JSON"
    OUTPUT_DIR = f"output"

    masked_codes_dict = json.load(open(f"{DATA_DIR}/testset_masked_source_codet5_truncated.json", "r"))
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

    infer_model = InferenceModel(
        GenerationModel(
            model_name="Salesforce/codet5-base",
            ckpt_path=GENERATION_CKPT_PATH,
            device="cuda"
        ),
        SimilarityModel(
            model_name="Salesforce/codet5-base",
            ckpt_path=SIMILARITY_CKPT_PATH,
            embedding_mode="avg",
            device="cuda"
        )
    )
    
    start = time.time()
    generating_list, ranking_list, cands_list, likelihoods_list, similarities_list = infer_model.infer(
        masked_codes,
        user_types_list,
        alpha=0.5,
        gen_k=5,
        gen_max_len=30,
        gen_batch_size=1,
        rank_batch_size=16
    )
    print(f"average inference time: {(time.time() - start) / len(ids)}")
    predictions = {
        id: {
            "generating": gens,
            "ranking": ranks,
            "candidates": cands,
            "likelihoods": likes,
            "similarities": sims
        } 
        for id, gens, ranks, cands, likes, sims in zip(ids, generating_list, ranking_list, cands_list, likelihoods_list, similarities_list)
    }
    Path(f"{OUTPUT_DIR}/predictions").mkdir(parents=True, exist_ok=True)
    json.dump(predictions, open(f"{OUTPUT_DIR}/predictions/randomsampled.json", "w"), indent=4)