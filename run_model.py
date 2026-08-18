"""
训练并评估单一模型的脚本
"""

import argparse

from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=False, help='whether save the trained model')
    parser.add_argument('--load_best_epoch', type=str2bool,
                        default=False, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--input_window', type=int, default=12, help='history length')
    parser.add_argument('--output_window', type=int, default=12, help='horizon length')
    parser.add_argument('--cache_dataset', type=str2bool, default=True, help='whether to load cache datatset')

    parser.add_argument('--hop', type=int, default=1, help='hops from the center road')
    parser.add_argument('--slot', type=int, default=1, help='times multiplied by 5min for each road')

    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--slice_length', type=int, default=4,
                        help='DynAggrSTGNN time steps to integrate another graph')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='DynAggrSTGNN GNN component layers')
    parser.add_argument('--similarity_threshold', type=float, default=0.99, help='DynAggrSTGNN similarity integration')
    parser.add_argument('--congestion_threshold', type=float, default=0.5,
                        help='DynAggrSTGNN congestion blocking integration')

    parser.add_argument('--subgraph_hops', type=int, default=2, help='Subgraph-variant hops of STID and STAEformer')
    parser.add_argument('--subgraph_batch_size', type=int, default=2,
                        help='Subgraph-variant sbs of STID and STAEformer')

    parser.add_argument('--exp_des', type=str, default=None, help='additional description of experiment')
    parser.add_argument('--use_early_stop', type=str2bool,
                        default=False, help='whether to use early stop')

    parser.add_argument('--model_des', type=str, default=None, help='additional description of model')



    parser.add_argument('--num_block', type=int, default=1, help='num of ST blocks')
    parser.add_argument('--tf_layers', type=int, default=3, help='num of tempX layers')
    parser.add_argument('--num_spa_heads', type=int, default=8, help='num of spaX multi-attn heads')
    parser.add_argument('--num_temp_heads', type=int, default=8, help='num of tempX multi-attn heads')
    parser.add_argument('--x_hdim', type=int, default=16, help='num of spaX hidden dimension')

    parser.add_argument('--input_embedding_dim', type=int, default=24, help='num of input dimension')
    parser.add_argument('--road_feature_embedding_dim', type=int, default=80, help='num of road feature embedding dimension')
    parser.add_argument('--tod_embedding_dim', type=int, default=24, help='num of tod dimension')
    parser.add_argument('--mhgat_heads', type=int, default=4, help='num of mhgat head')

    parser.add_argument('--train_loss', type=str, default="none", help='train_loss')

    parser.add_argument('--use_rf', type=str2bool, default=True, help='use road features as additional inputs')
    parser.add_argument('--use_trajDist', type=str2bool, default=True, help='use trajDist as additional inputs')
    parser.add_argument('--use_flowCnt', type=str2bool, default=True, help='use flowCnt as additional inputs')

    parser.add_argument('--use_mhgat', type=str2bool, default=True, help='use Multi-head GAT')
    parser.add_argument('--use_gcn', type=str2bool, default=True, help='use Classical GCN')
    parser.add_argument('--use_spaAttn', type=str2bool, default=True, help='use spatial attention')
    parser.add_argument('--use_tempAttn', type=str2bool, default=True, help='use temporal attention')


    parser.add_argument('--use_rmsNorm', type=str2bool, default=True, help='use RMSNorm instead of LayerNorm')
    parser.add_argument('--use_skip', type=str2bool, default=True, help='use Skip Connections')
    parser.add_argument('--use_residual', type=str2bool, default=True, help='use Residual Connections')

    parser.add_argument('--node_emb_path', type=str, default="./cache/road_representation_futian_1km_roadmap_edge_12_12_MRDVVGAE/evaluate_cache/embedding_MRDVVGAE_futian_1km_roadmap_edge_32.npy", help='node_embedding_path')

    # ==================== MRDVVGAE 消融实验开关（核心） ====================
    parser.add_argument('--ablation_use_struct_pattern', type=str2bool, default=True, help='是否使用 GIN 提取的结构模式特征')
    parser.add_argument('--ablation_use_func_context', type=str2bool, default=True, help='是否使用 LDA 推断的区域功能上下文')
    parser.add_argument('--ablation_use_multi_rel', type=str2bool, default=True, help='是否使用多关系逻辑视图公交共线关系')
    parser.add_argument('--ablation_use_bus_relation', type=str2bool, default=True, help='是否使用公交共线关系')
    parser.add_argument('--ablation_use_taxi_relation', type=str2bool, default=True, help='是否使用出租车转移关系')
    parser.add_argument('--ablation_use_attr_recon', type=str2bool, default=True, help='是否使用属性重构分支')
    parser.add_argument('--ablation_use_learnable_fusion', type=str2bool, default=True, help='是否使用可学习的 𝜶 进行融合')

    # ==================== EHITFormer消融实验开关（核心） ====================
    parser.add_argument('--use_missing_aware', type=str2bool, default=True, help='是否使用缺失感知嵌入')
    parser.add_argument('--use_road_emb', type=str2bool, default=True, help='是否使用路网异质嵌入')
    parser.add_argument('--use_dual_view', type=str2bool, default=True, help='是否使用双视图空间注意力')
    parser.add_argument('--use_logical_adj', type=str2bool, default=False, help='是否使用逻辑邻接矩阵')
    parser.add_argument('--use_gating', type=str2bool, default=True, help='是否使用门控融合（False=平均融合）')
    parser.add_argument('--use_confidence', type=str2bool, default=True, help='是否使用观测置信度加权')
    parser.add_argument('--use_temporal', type=str2bool, default=True, help='是否使用时间Transformer（False=纯空间）')
    parser.add_argument('--use_reconstruction', type=str2bool, default=True, help='是否使用重构分支（多任务）')
    parser.add_argument('--mask_in_input', type=str2bool, default=False, help='输入是否包含mask维度')


    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
                  val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
