import argparse

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t, load_out_emb_t
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import distill_run_transductive, distill_run_inductive
import networkx as nx
from position_encoding import DeepWalk
import dgl


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=".conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Student model number of layers"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Student model hidden layer dimensions",
    )
    # parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""

    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )

    """Distiall"""
    parser.add_argument(
        "--lamb",
        type=float,
        default=1,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    parser.add_argument(
        "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs"
    )

    # add-up
    parser.add_argument(
        "--dw",
        action="store_true",
        help="Set to True to include deepwalk positional encoding",
    )
    parser.add_argument(
        "--feat_distill",
        action="store_true",
        help="Set to True to include feature distillation loss",
    )
    parser.add_argument(
        "--adv",
        action="store_true",
        help="Set to True to include adversarial feature learning",
    )

    """parameter sensitivity"""
    parser.add_argument(
        "--sensitivity_adv_eps",
        type=float,
        default=-1,
        help="adv_eps for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_dw_emb_size",
        type=int,
        default=-1,
        help="dw_emb_size for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_feat_distill_weight",
        type=float,
        default=-1,
        help="feat_distill_weight for parameter sensitivity",
    )

    args = parser.parse_args()
    return args


global_trans_dw_feature = None


def get_features_dw(adj, device, is_transductive, args):
    if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
        print('getting dw for ogbn-arxiv/ogbn-products ...')
        G = adj
    else:
        adj = np.asarray(adj.cpu())
        G = nx.Graph(adj)

    model_emb = DeepWalk(G, walk_length=args.dw_walk_length, num_walks=args.dw_num_walks, workers=1)
    model_emb.train(window_size=args.dw_window_size, iter=args.dw_iter, embed_size=args.dw_emb_size)

    emb = model_emb.get_embeddings()  # get embedding vectors
    embeddings = []
    for i in range(len(emb)):
        embeddings.append(emb[i])
    embeddings = np.array(embeddings)
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    if is_transductive:
        global global_trans_dw_feature
        global_trans_dw_feature = embeddings
    else:  # inductive
        pass  # we don't have global_ind_dw_feature since each time seed (data split) is different.
    return embeddings


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0:
        if "noisy_features" not in str(args.output_path):
            args.output_path = Path.cwd().joinpath(
                args.output_path, "noisy_features", f"noise_{args.feature_noise}"
            )
        # Teacher is assumed to be trained on the same noisy features as well.
        # args.out_t_path = args.output_path

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        dw_emb_path = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            # "dw_emb.pt"
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )

    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )

    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    g = g.to(device)
    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1

    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(
            # args.model_config_path, args.student, args.dataset
            args.exp_setting + args.model_config_path, args.student, args.dataset
        )  # Note: student config
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")
    # print('conf: ', conf)

    # use parameters from conf
    if 'dw_walk_length' in conf and 'dw_walk_length' not in args:
        args.dw_walk_length = conf['dw_walk_length']
    if 'dw_num_walks' in conf and 'dw_num_walks' not in args:
        args.dw_num_walks = conf['dw_num_walks']
    if 'dw_window_size' in conf and 'dw_window_size' not in args:
        args.dw_window_size = conf['dw_window_size']
    if 'dw_iter' in conf and 'dw_iter' not in args:
        args.dw_iter = conf['dw_iter']
    if 'dw_emb_size' in conf and 'dw_emb_size' not in args:
        args.dw_emb_size = conf['dw_emb_size']
    if args.adv and 'adv_eps' in conf and 'adv_eps' not in args:
        args.adv_eps = conf['adv_eps']
    if args.feat_distill and 'feat_distill_weight' in conf and 'feat_distill_weight' not in args:
        args.feat_distill_weight = conf['feat_distill_weight']

    # parameter sensitivity
    if args.adv and args.sensitivity_adv_eps > 0:
        args.adv_eps = args.sensitivity_adv_eps
    if args.dw and args.sensitivity_dw_emb_size > 0:
        args.dw_emb_size = args.sensitivity_dw_emb_size
    if args.feat_distill and args.sensitivity_feat_distill_weight > 0:
        args.feat_distill_weight = args.sensitivity_feat_distill_weight

    len_position_feature = 0
    if args.exp_setting == "tran":
        idx_l = idx_train
        idx_t = torch.cat([idx_train, idx_val, idx_test])
        distill_indices = (idx_l, idx_t, idx_val, idx_test)

        # position feature (tran)
        if args.dw:
            if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
                dw_emb_path = dw_emb_path.joinpath("dw_emb.pt")
                try:
                    loaded_dw_emb = torch.load(dw_emb_path).to(device)
                    print('load dw_emb successfully!', flush=True)
                    position_feature = loaded_dw_emb
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                except:
                    print('cannot load dw_emb, now try to calculate it ...... ', flush=True)
                    network_g = g.cpu()
                    network_g = network_g.to_networkx()
                    print('done with network_g')
                    dw_emb = get_features_dw(network_g, device, is_transductive=True, args=args)
                    torch.save(dw_emb, dw_emb_path)
                    print('save dw_emb successfully')
                    position_feature = global_trans_dw_feature
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)

            # cpf datasets
            else:
                if args.cal_dw_flag:
                    adj = g.adj().to_dense()
                    get_features_dw(adj, device, is_transductive=True, args=args)

                position_feature = global_trans_dw_feature
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

    elif args.exp_setting == "ind":
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # position feature (ind)
        if args.dw:  # We need to run it every time since seed (data split) is different.
            # computation optimized for large datasets.
            if args.dataset == 'ogbn-products':
                dw_emb_path = output_dir.joinpath("dw_emb.pt")  # need to include the seed in the path
                # subgraph
                trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                network_g = trained_grapah.cpu()
                network_g = network_g.to_networkx()
                # print('done with network_g')
                position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                torch.save(position_feature_obs, dw_emb_path)
                # print('save dw_emb successfully')
                position_feature_obs = position_feature_obs.cpu()

                # change the order of position_feature_obs
                idx_position_feature = idx_obs.tolist()
                position_feature_list_correct_order = [[] for i in range(len(g.adj()))]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # get the neighbor for every node
                src_node, dst_node = g.edges()
                src_node = src_node.cpu().tolist()
                dst_node = dst_node.cpu().tolist()
                assert len(src_node) == len(dst_node)
                idx_test_ind_neighbor_dict = {}
                idx_test_ind_list = idx_test_ind.tolist()
                for i in range(len(src_node)):
                    src_node_i = src_node[i]
                    dst_node_i = dst_node[i]
                    if src_node_i not in idx_test_ind_neighbor_dict:
                        idx_test_ind_neighbor_dict[src_node_i] = []
                    idx_test_ind_neighbor_dict[src_node_i].append(dst_node_i)
                    if dst_node_i not in idx_test_ind_neighbor_dict:
                        idx_test_ind_neighbor_dict[dst_node_i] = []
                    idx_test_ind_neighbor_dict[dst_node_i].append(src_node_i)

                # get the dw for test nodes
                for idx_cur_node_id in idx_test_ind_list:
                    try:
                        idx_cur_node_id_neighbor = idx_test_ind_neighbor_dict[idx_cur_node_id]
                        if len(idx_cur_node_id_neighbor):
                            temp_position_feature = torch.mean(position_feature_obs[idx_cur_node_id_neighbor, :], dim=0)
                        else:
                            temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    except:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])

                    position_feature_obs[idx_cur_node_id] = torch.tensor(temp_position_feature, dtype=torch.float32)

                position_feature = position_feature_obs.to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)
                del position_feature_obs, position_feature  # save memory

            # not computation-friendly for large datasets (e.g., ogbn-products).
            elif args.dataset == 'ogbn-arxiv':
                dw_emb_path = output_dir.joinpath("dw_emb.pt")  # include the seed in the path

                # subgraph
                trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                network_g = trained_grapah.cpu()
                network_g = network_g.to_networkx()
                # print('done with network_g')
                position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                torch.save(position_feature_obs, dw_emb_path)
                # print('save dw_emb successfully')
                position_feature_obs = position_feature_obs.cpu()

                # change the order of position_feature_obs
                idx_position_feature = idx_obs.tolist()
                position_feature_list_correct_order = [[] for i in range(len(g.adj()))]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):  # tqdm(
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # get the dw for test nodes
                for idx_cur_node_id in idx_test_ind.tolist():  # tqdm(
                    temp_position_feature = None
                    counter_neighbor_in_obs = 0
                    _, idx_one_in_cur_node = g.out_edges(idx_cur_node_id)
                    idx_one_in_cur_node = idx_one_in_cur_node.tolist()
                    for idx_j in idx_one_in_cur_node:
                        if idx_j not in idx_position_feature:
                            continue
                        if temp_position_feature is None:
                            temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                        else:
                            temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                        counter_neighbor_in_obs += 1
                    # for those we could not find a neighbor
                    if temp_position_feature is None:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    else:
                        temp_position_feature /= counter_neighbor_in_obs
                    position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

            # cpf dataset
            else:
                adj = g.adj().to_dense()
                adj_obs = adj[idx_obs, :][:, idx_obs]

                # take dw from neighbors
                position_feature_obs = get_features_dw(adj_obs, device, is_transductive=False, args=args).cpu()

                idx_position_feature = idx_obs.tolist()
                # change the order of position_feature_obs
                position_feature_list_correct_order = [[] for i in range(len(adj))]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # fill in the dw for test nodes
                adj_numpy = adj.cpu().numpy()
                for idx_cur_node_id in idx_test_ind.tolist():
                    temp_position_feature = None
                    counter_neighbor_in_obs = 0
                    idx_one_in_cur_node = np.where(adj_numpy[idx_cur_node_id] == 1)[0]
                    idx_one_in_cur_node = idx_one_in_cur_node.tolist()
                    for idx_j in idx_one_in_cur_node:
                        if idx_j not in idx_position_feature:
                            continue
                        if temp_position_feature is None:
                            temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                        else:
                            temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                        counter_neighbor_in_obs += 1
                    # for those we could not find a neighbor
                    if temp_position_feature is None:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    else:
                        temp_position_feature /= counter_neighbor_in_obs
                    position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

    """ Model init """
    model = Model(conf, args, len_position_feature)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(conf["dataset"])

    """Load teacher model output"""
    out_t = load_out_t(out_t_dir)
    out_emb_t = load_out_emb_t(out_t_dir)
    out_emb_t = out_emb_t.to(device)
    logger.info(
        f"teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}"
    )
    logger.info(
        f"teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}"
    )
    logger.info(
        f"teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}"
    )

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":
        out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            g,
            args
        )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        out, score_val, score_test_tran, score_test_ind = distill_run_inductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            args
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving student outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut, flush=True)

    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        if seed == 0:
            cal_dw_flag = True
        else:
            cal_dw_flag = False
        args.cal_dw_flag = cal_dw_flag
        args.seed = seed
        scores.append(run(args))

    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        args.cal_dw_flag = True
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])
        if args.exp_setting == 'ind':
            score_prod = score[0] * 0.8 + score[1] * 0.2

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )
        if args.exp_setting == 'ind':
            score_prod = score_mean[0] * 0.8 + score_mean[1] * 0.2

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str, flush=True)
    if args.exp_setting == 'ind':
        print('prod: ', score_prod)


if __name__ == "__main__":
    args = get_args()
    main()
