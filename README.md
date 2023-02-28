
# NOSMOG
This repo contains code for the following two papers:
1. ICLR 2023 Spotlight paper "[Learning MLPs on Graphs: A Unified View of Effectiveness, Robustness, and Efficiency](https://openreview.net/forum?id=Cs3r5KLdoj)".
2. NeurIPS 2022 New Frontiers in Graph Learning paper "[NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs](https://arxiv.org/abs/2208.10010)".


## Preparing datasets
To run experiments for dataset used in the paper, please download from the following links and put them under `data/` (see below for instructions on organizing the datasets).

- *CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

- *OGB data* (`ogbn-arxiv` and `ogbn-products`): Datasets will be automatically downloaded when running the `load_data` function in `dataloader.py`. More details [here](https://ogb.stanford.edu/).


## Usage
To quickly train teacher models you can run `train_teacher.py` by specifying the experiment setting, i.e. transductive (`tran`) or inductive (`ind`), teacher model, e.g. `GCN`, and dataset, e.g. `cora`, as per the example below.

```
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora --num_exp 10 --max_epoch 200 --patience 50 --device 0
```

To quickly train student models with a pretrained teacher you can run `train_student.py` by specifying the experiment setting, teacher model, student model, and dataset like the example below. Make sure you train the teacher using the `train_teacher.py` first and have its result stored in the correct path specified by `--out_t_path`.

```
python train_student.py --exp_setting tran --teacher SAGE --student MLP --dataset citeseer --out_t_path outputs --num_exp 10 --max_epoch 200 --patience 50 --device 0 --dw --feat_distill --adv
```

For more examples, and to reproduce results in the paper, please refer to scripts in the folder `experiments/`. For example,

```
# train teacher GraphSAGE on five cpf datasets.
bash experiments/sage_cpf.sh

# train student NOSMOG on five cpf datasets.
bash experiments/nosmog_cpf.sh
```


To extend NOSMOG to your own model, you may do one of the following.
- Add your favourite model architectures to the `Model` class in `model.py`. Then follow the examples above.
- Train teacher model and store its output (log-probabilities). Then train the student by `train_student.py` with the correct `--out_t_path`.






## Citing NOSMOG

If you find NOSMOG useful, please cite our papers.
```
@inproceedings{NOSMOG_ICLR,
    title={Learning {MLP}s on Graphs: A Unified View of Effectiveness, Robustness, and Efficiency},
    author={Yijun Tian and Chuxu Zhang and Zhichun Guo and Xiangliang Zhang and Nitesh Chawla},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=Cs3r5KLdoj}
}

@inproceedings{NOSMOG_GLFrontiers,
    title={NOSMOG: Learning noise-robust and structure-aware mlps on graphs},
    author={Tian, Yijun and Zhang, Chuxu and Guo, Zhichun and Zhang, Xiangliang and Chawla, Nitesh V},
    booktitle={NeurIPS 2022 Workshop: New Frontiers in Graph Learning},
    year={2022},
    url={https://openreview.net/forum?id=nT897hw-hHD}
}

```