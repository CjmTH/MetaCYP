import numpy as np
import pandas as pd
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm, trange
from threading import Thread, Lock
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    jaccard_score, balanced_accuracy_score, matthews_corrcoef, f1_score
from rdkit import Chem
from rdkit.Chem import AllChem
import esm
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
import d2l.torch
from sklearn.preprocessing import label_binarize

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    th.cuda.manual_seed(random_seed)
    th.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.enabled = False


def calculate_molecule_3D_structure():
    def get_smiles_list_():
        data_df = pd.read_csv("/data/cjm/human_BoM/react_three_class_data/raw/ReactType_total.csv")
        smiles_list = data_df["molecule"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

    def calculate_molecule_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open('/data/cjm/human_BoM/react_three_class_data/result/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open('/data/cjm/human_BoM/react_three_class_data/result/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()

            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()

    mutex = Lock()
    os.system('rm /data/cjm/human_BoM/react_three_class_data/result/invalid_smiles.txt')
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict, open('/data/cjm/human_BoM/react_three_class_data/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))


def construct_data_list():
    data_df = pd.read_csv("/data/cjm/human_BoM/react_three_class_data/raw/ReactType_total.csv")
    smiles_to_conformation_dict = pkl.load(open('/data/cjm/human_BoM/react_three_class_data/intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["molecule"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "sequence": row["protein_seq"],
                "label": row["label"],
                "dataset_type": row["dataset_type"],
                #"pocket_site": row["protein_pocket"],
                #"SRS_site": row["protein_SRS"],
                "atom1_idx": row["atom1_idx"],
                "atom2_idx": row["atom2_idx"],
            }
            data_list.append(data_item)
    pkl.dump(data_list, open('/data/cjm/human_BoM/react_three_class_data/intermediate/data_list.pkl', 'wb'))

def convert_data_list_to_data_loader():
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('/data/cjm/human_BoM/react_three_class_data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        sequence_dataset = KeyDataset(data_list, "sequence")
        label_dataset = KeyDataset(data_list, "label")
       # pocket_site_dataset = KeyDataset(data_list, "pocket_site")
        #srs_site_dataset = KeyDataset(data_list, "SRS_site")
        atom1_index_dataset = KeyDataset(data_list, "atom1_idx")
        atom2_index_dataset = KeyDataset(data_list, "atom2_idx")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", False, False)  # 原来为第一个为True, 表示是否删除氢原子
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
                "smiles": RawArrayDataset(smiles_dataset),
                "sequence": RawArrayDataset(sequence_dataset),
                #"pocket_site": RawArrayDataset(pocket_site_dataset),
                #"SRS_site": RawArrayDataset(srs_site_dataset),
                "atom1_index": RawArrayDataset(atom1_index_dataset),
                "atom2_index": RawArrayDataset(atom2_index_dataset),
            },
            "target": {
                "label": RawLabelDataset(label_dataset),
            }
        })

    batch_size = 8  # 原来为8
    data_list = pkl.load(open('/data/cjm/human_BoM/react_three_class_data/intermediate/data_list.pkl', 'rb'))
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "valid"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    dataset_test = convert_data_list_to_dataset_(data_list_test)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_train.collater)
    data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_validate.collater)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset_test.collater)
    return data_loader_train, data_loader_valid, data_loader_test

class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('/data/cjm/human_BoM/react_three_class_data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(
            self,
            sample,
    ):
        net_input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type = net_input['src_tokens'], net_input['src_distance'], \
                                                             net_input['src_coord'], net_input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep[:, 1:, :],  # remove cls token
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output

# 交叉注意力
class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_cross_attention_1 = nn.LayerNorm(1280)
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=1280, vdim=1280, batch_first=True)
        self.layer_normalization_cross_attention_2 = nn.LayerNorm(512)
        self.feed_forward_cross_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, y, padding_mask):
        y = self.layer_normalization_cross_attention_1(y)
        y, _ = self.cross_attention(x, y, y, key_padding_mask=padding_mask)
        y_old = y
        y = self.layer_normalization_cross_attention_2(y)
        y = self.feed_forward_cross_attention(y)
        y = y + y_old
        return y

# 自注意力层
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_self_attention_1 = nn.LayerNorm(512)
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=512, vdim=512, batch_first=True)
        self.layer_normalization_self_attention_2 = nn.LayerNorm(512)
        self.feed_forward_self_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        x_old = x
        x = self.layer_normalization_self_attention_1(x)
        x, _ = self.self_attention(x, x, x)
        x = x + x_old
        x_old = x
        x = self.layer_normalization_self_attention_2(x)
        x = self.feed_forward_self_attention(x)
        x = x + x_old
        return x

class EsmUnimolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.molecule_encoder = UniMolModel()
        self.molecule_encoder.load_state_dict(th.load('/data/cjm/human_BoM/main/mol_pre_all_h_220816.pt')['model'], strict=False)
        self.protein_encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.protein_encoder.load_state_dict(th.load("/data/cjm/reaction_site/main/PF00067_0.9.pth"))
        self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=2048)

        self.transformer_layer_cross_attention = TransformerDecoderLayer()
        self.transformer_layer_self_attention = TransformerEncoderLayer()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3), #输出类别数5，对应5分类
        )

    def move_data_batch_to_cuda(self, data_batch):
        data_batch['input'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['input'].items()}
        data_batch['target'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['target'].items()}
        return data_batch

    def forward(self, data_batch):
        data_batch = self.move_data_batch_to_cuda(data_batch)

        # smiles_batch = data_batch['input']['smiles']
        molecule_encoder_output = self.molecule_encoder(data_batch)
        molecule_embedding = molecule_encoder_output['molecule_embedding']
        molecule_embedding_cls = molecule_encoder_output['molecule_representation']  # 取cls token作为分子的表示
        # print("molecule_embedding_cls:", molecule_embedding_cls.shape)#每个分子的embedding为512维，所以这里的shape为(分子个数, 512)
        molecule_smiles = molecule_encoder_output['smiles']
        # print("molecule_smiles:", molecule_smiles)
        # print("molecule_embedding:", molecule_embedding.shape)
        # print("molecule_representation:", molecule_embedding)
        atom1_index_batch = data_batch['input']['atom1_index'] #表示每个键连接的第一个原子的索引
        atom2_index_batch = data_batch['input']['atom2_index'] #表示每个键连接的第二个原子的索引
        # print("atom1_index_batch:", atom1_index_batch) #shape为(分子个数,)
        # print("atom2_index_batch:", atom2_index_batch)

        # 此时，每个键有两个原子，原子索引号分别为atom1_index和atom2_index，需要将每个原子的embedding取出来，并纵向拼接
        atom1_embedding = molecule_embedding[
            th.arange(molecule_embedding.size(0)), atom1_index_batch]  # molecule_embedding已经去掉了cls token，所以这里不需要+1

        # 当反应只有1个原子时，数据集中atom2_index标记为1000，此时需要将atom2_embedding设为全0向量，如果包含两个原子，则取对应的embedding
        for i in range(len(atom2_index_batch)):
            if atom2_index_batch[i] == 1000:
                atom2_index_batch[i] = -1  # 将1000替换为-1，方便后续处理,因为molecule_embedding的索引从0开始，到N-1结束，-1表示最后一个位置之外的位置，为什么不能是1000，因为molecule_embedding的shape为(分子个数, 原子个数, 512)，索引不能超过原子个数-1，但是-1可以用来表示无效位置
                #为什么1000不能用来表示无效位置，这里只是用来判断，并不会真的去取索引为1000的位置，因为molecule_embedding的shape远小于1000，会报错索引越界
        atom2_embedding = th.where(
            atom2_index_batch.unsqueeze(1) == -1,
            th.zeros_like(molecule_embedding[:, 0, :]),
            molecule_embedding[th.arange(molecule_embedding.size(0)), atom2_index_batch])
         #改行代码实现了根据atom2_index_batch的值来选择对应的embedding或全0向量，当atom2_index_batch的值为1000时，选择全0向量，否则选择对应的embedding。

        # print("atom_embedding:", atom_embedding.shape) #每个原子的embedding为512维，所以这里的shape为(原子个数, 512)
        # print("atom_embeding:", atom_embedding)
        # 将每个分子的cls token和原子的embedding拼接，作为每个分子中原子的embedding
        bond_molecule_embedding = []
        # print("len(atom_embedding):", len(atom_embedding))
        for i in range(len(atom1_embedding)):
            # print("i:", i)
            # cls = molecule_embedding_cls[i].reshape(1, -1)  # 将cls token的embedding转为(1, 512)
            # print("cls:", cls.shape)
            molecule_embedding_i = molecule_embedding[i]
            atom1 = atom1_embedding[i].reshape(1, -1)
            atom2 = atom2_embedding[i].reshape(1, -1)
            bond = th.cat((atom1, atom2), dim=0) # 将两个原子的embedding拼接,表示键的embedding
            # print("atom:", atom.shape)
            bond_molecule_embedding.append(th.cat((molecule_embedding_i, bond), dim=0)) # 将分子的embedding和键的embedding拼接，表示包含键信息的分子embedding，shape为(原子个数+2, 512)
        bond_molecule_embedding = th.stack(bond_molecule_embedding) # 最终bond_molecule_embedding的shape为(分子个数, 3, 512)

        # print("bond_molecule_embedding:", bond_molecule_embedding.shape) #最终每个batch的embedding的shape为(batch,3,512)
        # print("bond_molecule_embedding:", bond_molecule_embedding)

        sequence_batch = data_batch['input']['sequence']
        sequence_batch = [('', sequence) for sequence in sequence_batch]
        _, sequence_batch, token_batch = self.batch_converter(sequence_batch)
        token_batch = token_batch.cuda()
        protein_encoder_output = self.protein_encoder(token_batch, repr_layers=[33], return_contacts=False)
        protein_embedding = protein_encoder_output["representations"][33]

        x = self.transformer_layer_cross_attention(bond_molecule_embedding, protein_embedding, None)
        x = self.transformer_layer_self_attention(x)
        x = x[:, 0, :]
        x = self.mlp(x)
        return x

def evaluate(model, data_loader, csv_save):
    model.eval()
    label_predict = th.tensor([], dtype=th.float32).cuda()
    label_true = th.tensor([], dtype=th.long).cuda()
    with th.no_grad():
        for data_batch in data_loader:
            # for data_batch in tqdm(data_loader):
            label_predict_batch = model(data_batch)
            label_true_batch = data_batch['target']['label'].to(th.long)
            label_predict = th.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = th.cat((label_true, label_true_batch.detach()), dim=0)

    label_predict = th.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy() # 预测的类别概率分布，shape为(样本数, 5)
    predict_label = np.argmax(label_predict, axis=1) # 预测的类别标签，输出为0~4的整数
    label_true = label_true.cpu().numpy()

    # 将测试集预测结果和真实值保存到同一个csv文件
    if csv_save == True:
        df = pd.DataFrame(
            {'label_true': label_true, 'predict_label': predict_label})
        for c in range(label_predict.shape[1]):
            df[f"prob_class_{c}"] = label_predict[:, c] # 每个类别的预测概率

        df.to_csv('react_mcc_3_without_finetune_bs8.csv', index=False)
    # 对于多分类问题，计算AUC-ROC和AUC-PRC,ACC,MCC,Precision,Recall,F1-score,Jaccard
    try:
        auc_roc = roc_auc_score(label_true, label_predict, multi_class='ovr')
    except ValueError as e:
        print("AUC-ROC skipped:", e)
        auc_roc = np.nan

    # AUC-PRC（多分类，跳过无正样本类别）
    # 计算macro平均的AUC-PRC,对于多分类问题，average参数设置为'macro',表示计算每个类别的AUC-PRC，然后取平均值,不考虑类别不平衡问题
    ap_list = []
    num_classes = label_predict.shape[1]

    for c in range(num_classes):
        y_c = (label_true == c).astype(int) # 将真实标签转换为二进制格式，表示是否属于类别c,输出为0或1
        if y_c.sum() == 0:
            continue  # 测试集中该类不存在，跳过
        ap = average_precision_score(y_c, label_predict[:, c]) # 计算类别c的AUC-PRC
        ap_list.append(ap)

    auc_prc = np.mean(ap_list) if len(ap_list) > 0 else np.nan # 计算所有类别的AUC-PRC的平均值

    accuracy = accuracy_score(label_true, predict_label)
    balanced_accuracy = balanced_accuracy_score(label_true, predict_label)
    precision = precision_score(label_true, predict_label, average='macro', zero_division=0)
    recall = recall_score(label_true, predict_label, average='macro', zero_division=0)
    f1 = f1_score(label_true, predict_label, average='macro', zero_division=0)
    jaccard = jaccard_score(label_true, predict_label, average='macro', zero_division=0)
    mcc = matthews_corrcoef(label_true, predict_label)

    metric = {'auc_roc': auc_roc, 'auc_prc': auc_prc, 'accuracy': accuracy, "balanced_accuracy": balanced_accuracy,
              'precision': precision, 'recall': recall, 'f1': f1, "jaccard": jaccard, "mcc": mcc}
    return metric


def train(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader()
    dataset = data_loader_train.dataset  # 获取底层数据集

    model = EsmUnimolClassifier()
    model.cuda()

    # 冻结esm2模型的参数
    for param in model.protein_encoder.parameters():
        param.requires_grad = False


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = -1e10
    max_bearable_epoch = 30
    current_best_epoch = 0

    for epoch in range(2000):
        model.train()
        for step, data_batch in enumerate(data_loader_train):
            label_predict_batch = model(data_batch)
            label_true_batch = data_batch['target']['label'].to(th.long)
            # print(label_predict_batch)
            # print(label_true_batch)
            # print(label_predict_batch.shape, label_true_batch.shape)

            loss = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 2000 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, round(loss.item(), 3)))
            #     # 打印训练过程后12层网络层的权重,验证ESM2的权重是否冻结
            #     no_finetune_layer = [0, 33]
            #     print("step:", step)
            #     weight = {}
            #     for name, value in model.protein_encoder.named_parameters():
            #         for t in no_finetune_layer:
            #             if "layers.%d" % t in name:
            #                 if 'final_layer_norm.weight' in name:
            #                     weight[name] = value[:10]
            #     print(name, weight)  # 打印权重的前10个值

        scheduler.step()

        metric_train = evaluate(model, data_loader_train, csv_save=False)
        metric_validate = evaluate(model, data_loader_validate, csv_save=False)
        metric_test = evaluate(model, data_loader_test, csv_save=False)

        # 监督指标改为auc和ap的均值：(auc_roc + auc_prc) / 2
        if metric_validate['mcc']  >= current_best_metric:
            current_best_metric = metric_validate['mcc']
            current_best_epoch = epoch
            th.save(model.state_dict(), f"../weight/{trial_version}.pt")
        print("==================================================================================")
        print('Epoch', epoch)
        print('Train', metric_train)
        print('validate', metric_validate)
        print('Test', metric_test)
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        print("==================================================================================")
        if epoch > current_best_epoch + max_bearable_epoch:
            break


def test(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader()

    model = EsmUnimolClassifier()
    model.cuda()
    model.load_state_dict(th.load(f"../weight/{trial_version}.pt"))

    metric_train = evaluate(model, data_loader_train, csv_save=False)
    metric_validate = evaluate(model, data_loader_validate, csv_save=False)
    metric_test = evaluate(model, data_loader_test, csv_save=True)
    print("Train", metric_train)
    print("validate", metric_validate)
    print("Test", metric_test)


if __name__ == "__main__":
    set_random_seed(1024)
    # print("data_process start!")
    # calculate_molecule_3D_structure()
    # construct_data_list()

    print("train start!")
    train(trial_version='react_mcc_3_without_finetune_bs8')  # 表示训练集为全部训练集时的结果

    print("test start!")
    test(trial_version='react_mcc_3_without_finetune_bs8')


    print('All is well!')
