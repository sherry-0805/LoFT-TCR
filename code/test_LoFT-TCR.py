import torch
import torch.nn.functional as F
import torchmetrics
import pandas as pd
import os

from LoftTCR import *
from data_process import *
from construct_strutural_neighborhood import *
from config import *
from sklearn.metrics import precision_recall_curve, auc
import utils_functions
import umap
from sklearn.preprocessing import StandardScaler

# config
NUM_EPOCHS = int(args.epochs)
cuda_name = 'cuda:' + args.cuda

root_model = args.modeldir

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
data_test, test_edge_label_index, y_test = create_dataset_global_predict()
g_test,test_edge_label_index = process_data_to_homogeneous_graph(data_test, dim_reduction = args.dimReduction, consNeighborType = args.construct_neighbor)
g_test = g_test.to(device)
test_edge_label_index = test_edge_label_index.to(device)

if args.mode == 'binary':
    y_test = torch.tensor(y_test).float().to(device)
elif args.mode == 'trinary':
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
else:
    print("mode error")

def predict(model):
    print('Testing on {} samples...'.format(len(data_test['cdr3b', 'CBindA', 'peptide'].edge_index[0])))
    
    if args.mode == 'binary':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.mode == 'trinary':
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("mode error")

    ce_weight = 0.4
    tversky_weight = 0.6
    tversky_loss_fn = utils_functions.TverskyLoss(alpha=0.7, gamma=2, mode=args.mode)

    model.eval()

    with torch.no_grad():
        out_test, intermediate_x = model(g_test, g_test.ndata['feat'], test_edge_label_index)
        test_ce_loss = loss_fn(out_test, y_test)

        if args.mode == 'binary':
            test_preds = torch.sigmoid(out_test)
        elif args.mode == 'trinary':
            test_preds = F.softmax(out_test, dim=1)
        else:
            print("mode error")

        test_tversky_loss=tversky_loss_fn(test_preds,y_test)
        tversky_val = 1 - test_tversky_loss.item()
        test_loss = test_ce_loss  * ce_weight + test_tversky_loss * tversky_weight

        if args.mode == 'binary':
            test_accuracy = torchmetrics.functional.accuracy(test_preds, y_test.int(), task="binary")
            test_ROCAUC = torchmetrics.functional.auroc(test_preds, y_test.int(), task="binary")
            precision, recall, _ = precision_recall_curve(y_test.int().cpu().numpy(),test_preds.cpu().numpy())
            test_AUPRC = auc(recall, precision)
        elif args.mode == 'trinary':
            test_accuracy = torchmetrics.functional.accuracy(test_preds, y_test.int(),task="multiclass", num_classes=3,
                average='micro'
            )
            test_ROCAUC = torchmetrics.functional.auroc(test_preds, y_test.int(),task="multiclass",num_classes=3,
                average='macro'
            )
            precision_per_class = []
            recall_per_class = []
            for i in range(3):
                y_test_cpu = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
                test_preds_cpu = test_preds.cpu().numpy() if test_preds.is_cuda else test_preds.numpy()

                precision, recall, _ = precision_recall_curve(y_test_cpu == i, test_preds_cpu[:, i])
                precision_per_class.append(precision)
                recall_per_class.append(recall)
            AUPRC_per_class = [auc(recall, precision) for recall, precision in zip(recall_per_class, precision_per_class)]
            test_AUPRC = sum(AUPRC_per_class) / len(AUPRC_per_class)
        else:
            print("mode error")

    return test_loss, test_accuracy, test_ROCAUC, test_AUPRC, test_preds, intermediate_x, y_test.int()
def perform_umap(features,
                umap_n_neighbors=15, 
                umap_min_dist=0.1, 
                umap_metric='euclidean', 
                random_state=42):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("UMAP...")
    reducer = umap.UMAP(n_components=2, 
                            n_neighbors=umap_n_neighbors, 
                            min_dist=umap_min_dist, 
                            metric=umap_metric, 
                            random_state=random_state)
    umap_features = reducer.fit_transform(scaled_features)
    return umap_features

def umap_and_save_csv(features, test_cdr3b, test_peptide, y_test, path_csv):

    # UMAP
    umap_results = perform_umap(features)

    output_data = {
        'peptide': test_peptide,
        'cdr3': test_cdr3b,
        'Binding': y_test.cpu().numpy(),
        'UMAP_1': umap_results[:, 0], 
        'UMAP_2': umap_results[:, 1]
    }

    output_df = pd.DataFrame(output_data)

    # 保存
    output_df.to_csv(path_csv, index=False)
    print(f"CSV file with UMAP results has been saved at {path_csv}")


if __name__=="__main__":
    model = TridentTCR(
        mode=args.mode,
        num_input_features=g_test.ndata['feat'].size(1),
        num_hidden=args.num_hidden,
        num_output_classes=512,
        num_divisions=args.num_divisions,
        num_heads_layer_one=args.num_heads_layer_one,
        dropout_rate=args.dropout_rate,
        layer_one_ggcn_merge=args.layer_one_ggcn_merge,
        layer_one_channel_merge=args.layer_one_channel_merge,
        latent_weight=args.latent_weight,
        selfloop_weight=args.selfloop_weight
    )

    model = model.to(device)
    
    with torch.no_grad():  # Initialize lazy modules.
        out, intermediate_x = model(g_test, g_test.ndata['feat'], test_edge_label_index)
    
    model_dir = args.testmodeldir
    val_model_path = os.path.join(root_model, model_dir)
    for root, dirs, files in os.walk(val_model_path):
        for file in files:
            if file.startswith("min"):
                PATH = os.path.join(val_model_path, file)
                model.load_state_dict(torch.load(PATH))
                test_loss, test_accuracy, test_ROCAUC, test_AUPRC, test_prob, intermediate_x, y_test = predict(model)
                print("ACC: {:.4f}".format(test_accuracy))
                print("AUC: {:.4f}".format(test_ROCAUC))
                print("AUPRC: {:.4f}".format(test_AUPRC))

                root = os.path.join(args.pridir, args.secdir, args.terdir)
                test_data = pd.read_csv(os.path.join(root, 'test.tsv'), delimiter='\t')
                test_cdr3b, test_peptide = list(test_data['cdr3']), list(test_data['peptide'])
                if args.mode == 'binary':
                    df = pd.DataFrame({'cdr3': test_cdr3b, 'peptide': test_peptide, 'probability': test_prob.cpu()})
                elif args.mode == 'trinary':
                    df = pd.DataFrame({'cdr3': test_cdr3b, 'peptide': test_peptide, 'class_0_prob': test_prob.cpu()[:, 0], 'class_1_prob': test_prob.cpu()[:, 1], 'class_2_prob': test_prob.cpu()[:, 2]})
                else:
                    print("mode error")       
                # umap
                # umap_and_save_csv(intermediate_x.cpu().numpy(), test_cdr3b, test_peptide, y_test, os.path.join(root, "TridentTCR_umap.csv"))
                path_str = "pred.tsv"
                df.to_csv(os.path.join(root, path_str), header=True, sep='\t', index=False)
