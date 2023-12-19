import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from hityper.typeobject import TypeObject

EXACT = True
# EXACT = False

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "output/"

    TESTSET_VERSION = "randomsampled"
    # TESTSET_VERSION = "Overall"


    Path(f"{OUTPUT_DIR}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/logs").mkdir(parents=True, exist_ok=True)

    masked_code_dict = json.load(open(f"{DATA_DIR}/testset_masked_source_codet5.json", "r"))
    user_types_dict = json.load(open(f"{DATA_DIR}/testset_usertypes.json", "r"))
    training_types = {info[1] for _, info in json.load(open(f"{DATA_DIR}/trainset_transformed.json", "r")).items()}

    STEP = 0.1
    ALPHA = [round(alpha * STEP, 2) for alpha in list(range(int(1 / STEP) + 1))]
    print(ALPHA)
    K = [1,2,3,4,5]

    POINTS = {
        "Overall": [],
        "Var": [],
        "Arg": [],
        "Ret": [],
        "Ele": [],
        "Gen": [],
        "Usr": [],
        "Seen": [],
        "Unseen": [],
    }

    def statistic(prediction_dict: dict, transformed_dict:dict, metrics_path, log_path, alpha):
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
            orig_preds, final_preds = prediction_dict[id]["original"], prediction_dict[id]["ranking"]

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
            for idx, (pred, score) in enumerate(orig_preds, 1):
#             for idx, pred in enumerate(orig_preds, 1):
#                 score = 0
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
            # lik_norm = sum(lik for _, _, lik, _ in rank_preds)
            lik_norm = 1.
            rank_preds = [(pred, sim / 2 + 0.5, lik / lik_norm) for pred, sim, lik, _ in final_preds]
            rank_preds = [(pred, sim, lik, alpha * lik + (1- alpha) * sim) for pred, sim, lik in rank_preds]
            rank_preds = [(pred, sim, lik, score) for pred, sim, lik, score in rank_preds if pred in user_types]
            rank_preds.sort(key=lambda pred: pred[-1], reverse=True)
            ranking = 999999
            for idx, (pred, sim, lik, score) in enumerate(rank_preds, 1):
                _logs.append(f"** idx: {idx}, pred type: {pred}, similarity: {sim}, likelihood: {lik}, score: {score}")
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
            log_infos.append("\n".join(_logs))
            
            _logs.append('----- OURS -----')
            # lik_norm = sum(lik for _, _, lik, _ in final_preds)
            lik_norm = 1.
            final_preds = [(pred, sim / 2 + 0.5, lik / lik_norm) for pred, sim, lik, _ in final_preds]
            final_preds = [(pred, sim, lik, alpha * lik + (1- alpha) * sim) for pred, sim, lik in final_preds]
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
                metrics_infos.append(f"top-{k}")
                metrics_infos.append(f"\t[generating]  cnt: {gen_topn_dict[kind][k]} acc: {gen_topn_dict[kind][k]/total_count if total_count else 0}")
                metrics_infos.append(f"\t[ ranking  ]  cnt: {rank_topn_dict[kind][k]} acc: {rank_topn_dict[kind][k]/total_count if total_count else 0}")
                metrics_infos.append(f"\t[   ours   ]  cnt: {ours_topn_dict[kind][k]} acc: {ours_topn_dict[kind][k]/total_count if total_count else 0}")
            metrics_infos.append("\n\n\n")

            POINTS[kind].append(ours_topn_dict[kind][5]/total_count * 100)
        open(metrics_path, "w").write("\n".join(metrics_infos))

        open(log_path, "w").write("\n\n\n".join(log_infos))

         
    prediction_dict = json.load(open(f"{OUTPUT_DIR}/predictions/{TESTSET_VERSION}.json", "r"))
    transformed_dict = json.load(open(f"{DATA_DIR}/testset_{TESTSET_VERSION}_transformed.json", "r"))
    
    exact = "" if EXACT else "-basematch"

    for alpha in ALPHA:
        statistic(
            prediction_dict,
            transformed_dict,
            f"{OUTPUT_DIR}/metrics/{TESTSET_VERSION}-alpha{alpha}{exact}.txt",
            f"{OUTPUT_DIR}/logs/{TESTSET_VERSION}-alpha{alpha}{exact}.txt",
            alpha
        )

    fig, ax = plt.subplots(1, 1)
    for kind, ys in POINTS.items():
        ax.plot(ALPHA, ys, label=kind)
    leg = ax.legend(loc="lower left", bbox_to_anchor=[0, 0.01],
                 ncols=3, shadow=False, title="Legend", fancybox=True)
    leg.get_title().set_color("red")
    plt.savefig(fname=f"{OUTPUT_DIR}/metrics/{TESTSET_VERSION}{exact}.jpg", format="jpg", dpi=600)
