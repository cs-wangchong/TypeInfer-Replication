import json
from pathlib import Path

from hityper.typeobject import TypeObject


EXACT = True
EXACT = False

if __name__ == "__main__":
    OUTPUT_DIR = "output"
    DATA_DIR = "data/ManyTypes4Py-JSON"
    TESTSET_VERSION = "randomsampled"

    Path(f"{OUTPUT_DIR}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/logs").mkdir(parents=True, exist_ok=True)

    masked_code_dict = json.load(open(f"{DATA_DIR}/testset_masked_source_codet5.json", "r"))
    user_types_dict = json.load(open(f"{DATA_DIR}/testset_usertypes.json", "r"))
    training_types = {info[1] for _, info in json.load(open(f"{DATA_DIR}/trainset_transformed.json", "r")).items()}

    K = [1, 3, 5]

    def statistic(prediction_dict: dict, transformed_dict:dict, metrics_path, log_path):
        count_dict = {
            "Overall": 0,
            "Var": 0,
            "Arg": 0,
            "Ret": 0,
            "Ele": 0,
            "Gen": 0,
            "Usr": 0,
            "Seen": 0,
            "Unseen": 0,
        }
        gen_topn_dict = {
            "Overall": {k:0 for k in K},
            "Var": {k:0 for k in K},
            "Arg": {k:0 for k in K},
            "Ret": {k:0 for k in K},
            "Ele": {k:0 for k in K},
            "Gen": {k:0 for k in K},
            "Usr": {k:0 for k in K},
            "Seen": {k:0 for k in K},
            "Unseen": {k:0 for k in K},
        }
        rank_topn_dict = {
            "Overall": {k:0 for k in K},
            "Var": {k:0 for k in K},
            "Arg": {k:0 for k in K},
            "Ret": {k:0 for k in K},
            "Ele": {k:0 for k in K},
            "Gen": {k:0 for k in K},
            "Usr": {k:0 for k in K},
            "Seen": {k:0 for k in K},
            "Unseen": {k:0 for k in K},
        }
        ours_topn_dict = {
            "Overall": {k:0 for k in K},
            "Var": {k:0 for k in K},
            "Arg": {k:0 for k in K},
            "Ret": {k:0 for k in K},
            "Ele": {k:0 for k in K},
            "Gen": {k:0 for k in K},
            "Usr": {k:0 for k in K},
            "Seen": {k:0 for k in K},
            "Unseen": {k:0 for k in K},
        }


        log_infos = []
        for id, (name, gt_type, src_kind) in transformed_dict.items():
            if id not in prediction_dict:
                continue

            masked_code, user_types = masked_code_dict[id], user_types_dict[id][1]
            gt_type_obj = TypeObject.Str2Obj(gt_type)
            generating_preds, ranking_preds, all_cands, likelihoods, similarities =\
                prediction_dict[id]["generating"], prediction_dict[id]["ranking"], prediction_dict[id]["candidates"], prediction_dict[id]["likelihoods"], prediction_dict[id]["similarities"]

            if src_kind.startswith("depth"):
                src_kind = "Gen"
            elif src_kind == "user-defined":
                src_kind = "Usr"
            else:
                src_kind = "Ele"

            pos_kind = "Var"
            if id.endswith("arg"):
                pos_kind = "Arg"
            elif id.endswith("return"):
                pos_kind = "Ret"
            
            seen_kind = "Unseen" if (gt_type not in training_types and src_kind == "Usr") else "Seen"

            count_dict["Overall"] += 1
            count_dict[src_kind] += 1
            count_dict[pos_kind] += 1
            count_dict[seen_kind] += 1

            _logs = []
            _logs.append('='* 50)
            _logs.append(id)
            _logs.append(masked_code)
            _logs.append(f'** name: {name}')
            _logs.append(f'** source kind: {src_kind}')
            _logs.append(f'** position kind: {pos_kind}')
            _logs.append(f'** seen kind: {seen_kind}')
            _logs.append(f'** user types: {user_types}')
            _logs.append(f'** groundtruth type: {gt_type}')

            _logs.append('----- ONLY GENERATING -----')
            ranking = 999999
            for idx, (pred, score) in enumerate(generating_preds, 1):       
                _logs.append(f"** idx: {idx}, pred type: {pred}, score: {score}")
                pred_type_obj = TypeObject.Str2Obj(pred)
                if (EXACT and TypeObject.isIdenticalSet(gt_type_obj, pred_type_obj)) or (not EXACT and TypeObject.isSetIncluded2(gt_type_obj, pred_type_obj)):
                    if idx < ranking:
                        ranking = idx
                        _logs.append(f"** hit!!!!")
            
            for k in K:
                if ranking <= k:
                    gen_topn_dict["Overall"][k] += 1
                    gen_topn_dict[src_kind][k] += 1
                    gen_topn_dict[pos_kind][k] += 1
                    gen_topn_dict[seen_kind][k] += 1

            _logs.append('----- ONLY RANKING -----')
            user_types = set(user_types)
            rank_preds = [(pred, sim) for pred, sim in zip(all_cands, similarities) if pred in user_types]
            rank_preds.sort(key=lambda pred: pred[-1], reverse=True)
            ranking = 999999
            for idx, (pred, sim) in enumerate(rank_preds, 1):
                _logs.append(f"** idx: {idx}, pred type: {pred}, similarity: {sim}")
                pred_type_obj = TypeObject.Str2Obj(pred)
                if (EXACT and TypeObject.isIdenticalSet(gt_type_obj, pred_type_obj)) or (not EXACT and TypeObject.isSetIncluded2(gt_type_obj, pred_type_obj)):
                    if idx < ranking:
                        ranking = idx
                        _logs.append(f"** hit!!!!")
            for k in K:
                if ranking <= k:
                    rank_topn_dict["Overall"][k] += 1
                    rank_topn_dict[src_kind][k] += 1
                    rank_topn_dict[pos_kind][k] += 1
                    rank_topn_dict[seen_kind][k] += 1
            # log_infos.append("\n".join(_logs))
            
            _logs.append('----- OURS -----')
            valid_set = set()
            for gen, _ in generating_preds:
                # base = gen.split("[")[0].split(".")[-1]
                # if base not in BUILTINS and base not in user_types:
                #     continue
                valid_set.add(gen)
            valid_set.update(user_types)
            _logs.append(f"valid cands: {valid_set}")
            # lik_norm = sum(lik for _, _, lik, _ in final_preds)
            lik_norm = 1.
            final_preds = [(pred, sim / 2 + 0.5, lik / lik_norm) for pred, lik, sim in zip(all_cands, likelihoods, similarities) if pred in valid_set]
            final_preds = [(pred, sim, lik, 0.5 * lik + 0.5 * sim) for pred, sim, lik in final_preds]
            final_preds.sort(key=lambda pred: pred[-1], reverse=True)
            ranking = 999999
            for idx, (pred, sim, lik, score) in enumerate(final_preds, 1):
                _logs.append(f"** idx: {idx}, pred type: {pred}, similarity: {sim}, likelihood: {lik}, score: {score}")
                pred_type_obj = TypeObject.Str2Obj(pred)
                if (EXACT and TypeObject.isIdenticalSet(gt_type_obj, pred_type_obj)) or (not EXACT and TypeObject.isSetIncluded2(gt_type_obj, pred_type_obj)):
                    if idx < ranking:
                        ranking = idx
                        _logs.append(f"** hit!!!!")
            for k in K:
                if ranking <= k:
                    ours_topn_dict["Overall"][k] += 1
                    ours_topn_dict[src_kind][k] += 1
                    ours_topn_dict[pos_kind][k] += 1
                    ours_topn_dict[seen_kind][k] += 1
            log_infos.append("\n".join(_logs))

        metrics_infos = []
        for kind, total_count in count_dict.items():
            metrics_infos.append(f'--------------- {kind.upper()} ---------------')
            metrics_infos.append(f"total count: {total_count}")
            for k in K:
                metrics_infos.append(f"##### TOP-{k} #####")
                metrics_infos.append(f"\t[only generating] acc: {gen_topn_dict[kind][k]/total_count if total_count else 0}")
                metrics_infos.append(f"\t[ only  ranking ] acc: {rank_topn_dict[kind][k]/total_count if total_count else 0}")
                metrics_infos.append(f"\t[     TIGER     ] acc: {ours_topn_dict[kind][k]/total_count if total_count else 0}")
            metrics_infos.append("\n\n\n")
        
        open(metrics_path, "w").write("\n".join(metrics_infos))
        open(log_path, "w").write("\n\n\n".join(log_infos))

    prediction_dict = json.load(open(f"{OUTPUT_DIR}/predictions/{TESTSET_VERSION}.json", "r"))
    transformed_dict = json.load(open(f"{DATA_DIR}/testset_{TESTSET_VERSION}_transformed.json", "r"))
    
    exact = "exact-match" if EXACT else "base-match"

    statistic(
        prediction_dict,
        transformed_dict,
        f"{OUTPUT_DIR}/metrics/{TESTSET_VERSION}-{exact}.txt",
        f"{OUTPUT_DIR}/logs/{TESTSET_VERSION}-{exact}.txt",
    )