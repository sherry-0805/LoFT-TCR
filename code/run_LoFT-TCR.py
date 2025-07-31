import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import pandas as pd
import os
import utils_functions
from data_process import *
from config import *
import logging
from LoftTCR import *
from construct_strutural_neighborhood import *
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, auc


# config
logging.basicConfig(filename='training_log.txt', level=logging.INFO)
NUM_EPOCHS = int(args.epochs)
cuda_name = 'cuda:' + args.cuda
root_model = args.modeldir
checkpoint_suffix = "lora"
model_dir = str(args.secdir) + '_' + str(args.terdir) + checkpoint_suffix
save_model_path = os.path.join(root_model, model_dir)
root_history = args.hisdir
best_auc_model_path = os.path.join(save_model_path, "best_auc_model.pth")

#print(model_dir)
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
data_train, data_test, train_edge_label_index, test_edge_label_index, y_train, y_test = create_dataset_global()


def train_with_autocast(model, ep, scaler=None, profiler=None):
    print('Training on {} samples...'.format(len(data_train['cdr3b', 'CBindA', 'peptide'].edge_index[0])))
    
    if args.mode == 'binary':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.mode == 'trinary':
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("mode error")
    ce_weight = 0.4
    tversky_weight = 0.6
    tversky_loss_fn = utils_functions.TverskyLoss(alpha=0.7, gamma=2, mode=args.mode)

    model.train()
    optimizer.zero_grad()


    with torch.cuda.amp.autocast():
        out, _ = model(g_train, g_train.ndata['feat'], train_edge_label_index)
        train_ce_loss = loss_fn(out, y_train)
        if args.mode == 'binary':
            train_preds = torch.sigmoid(out)
        elif args.mode == 'trinary':
            train_preds = F.softmax(out, dim=1)
        else:
            print("mode error")
        train_tversky_loss=tversky_loss_fn(train_preds,y_train)
        train_loss = train_ce_loss * ce_weight + train_tversky_loss * tversky_weight

    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    train_labels = y_train

    if args.mode == 'binary':
        train_accuracy = torchmetrics.functional.accuracy(train_preds, train_labels.int(), task="binary")
        train_ROCAUC = torchmetrics.functional.auroc(train_preds, train_labels.int(), task="binary")
        precision, recall, _ = precision_recall_curve(train_labels.int().cpu().numpy(),train_preds.detach().cpu().numpy())
        train_AUPRC = auc(recall, precision)
    elif args.mode == 'trinary':
        train_accuracy = torchmetrics.functional.accuracy(train_preds, train_labels,task="multiclass", num_classes=3,
            average='micro'
        )
        train_ROCAUC = torchmetrics.functional.auroc(train_preds, train_labels,task="multiclass",num_classes=3,
            average='macro'
        )
        precision_per_class = []
        recall_per_class = []
        for i in range(3):
            train_labels_cpu = train_labels.cpu().numpy() if train_labels.is_cuda else train_labels.numpy()
            train_preds_cpu = train_preds.detach().cpu().numpy() if train_preds.is_cuda else train_preds.detach().numpy()

            precision, recall, _ = precision_recall_curve(train_labels_cpu == i, train_preds_cpu[:, i])
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        AUPRC_per_class = [auc(recall, precision) for recall, precision in zip(recall_per_class, precision_per_class)]
        train_AUPRC = sum(AUPRC_per_class) / len(AUPRC_per_class) 
        
    else:
        print("mode error")
    
    model.eval()
    with torch.no_grad():
        with autocast():
            out_test, _ = model(g_test, g_test.ndata['feat'], test_edge_label_index)
            ce_test_loss = loss_fn(out_test, y_test)
            if args.mode == 'binary':               
                test_preds = torch.sigmoid(out_test)
            else:
                test_preds = F.softmax(out_test, dim=1)

            test_tversky_loss=tversky_loss_fn(test_preds,y_test)
            tversky_val = 1 - test_tversky_loss.item()
            print(f"test Tversky Index: {tversky_val:.4f}")
            test_loss = ce_test_loss * ce_weight + test_tversky_loss * tversky_weight

            test_labels = y_test
            if args.mode == 'binary':
                test_accuracy = torchmetrics.functional.accuracy(test_preds, test_labels.int(), task="binary")
                test_ROCAUC = torchmetrics.functional.auroc(test_preds, test_labels.int(), task="binary")
                precision, recall, _ = precision_recall_curve(y_test.int().cpu().numpy(),test_preds.cpu().numpy())
                test_AUPRC = auc(recall, precision)
            elif args.mode == 'trinary':
                test_accuracy = torchmetrics.functional.accuracy(test_preds, test_labels,task="multiclass", num_classes=3,
                    average='micro')
        
                test_ROCAUC = torchmetrics.functional.auroc(test_preds, test_labels,task="multiclass",num_classes=3,
                    average='macro')
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

        if ep % 100 == 0:
            if args.mode == 'binary':
                tp, tn, fp, fn, accuracy, precision, recall, f1 = utils_functions.calculate_binary_confusion_matrix(test_preds, y_test)
            else:
                tp, tn, fp, fn, accuracy, precision, recall, f1 =utils_functions.calculate_trinary_confusion_matrix(test_preds, y_test, num_classes=3)
            
            logging.info(
                f"Epoch {ep}: TP={tp}, TN={tn}, FP={fp}, FN={fn},Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    if profiler:
        profiler.step()
    return train_loss, train_accuracy, train_ROCAUC, train_AUPRC, test_loss, test_accuracy, test_ROCAUC, test_AUPRC

g_train,train_edge_label_index = process_data_to_homogeneous_graph(data_train, dim_reduction = args.dimReduction, consNeighborType = args.construct_neighbor)
g_test,test_edge_label_index = process_data_to_homogeneous_graph(data_test, dim_reduction = args.dimReduction, consNeighborType = args.construct_neighbor)
g_train, g_test = g_train.to(device), g_test.to(device)
train_edge_label_index, test_edge_label_index = train_edge_label_index.to(device), test_edge_label_index.to(device)

if args.mode == 'binary':
    y_train, y_test = torch.tensor(y_train).float().to(device), torch.tensor(y_test).float().to(device)
elif args.mode == 'trinary':
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
else:
    print("mode error")

print(f'Testing {args.dimReduction} dimensionality reduction')


if __name__ == "__main__":
    best_auc = -float("inf")
    best_auc_epoch = 0
    scaler = torch.cuda.amp.GradScaler()

    print("latent weight equals ",args.latent_weight)
    print("selfloop weight equals ",args.selfloop_weight)
    model = TridentTCR(
        mode=args.mode,
        num_input_features=g_train.ndata['feat'].size(1),
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
    with torch.no_grad():
        out,_ = model(g_train, g_train.ndata['feat'], train_edge_label_index)

    optimizer = torch.optim.Adam(
        [
            {'params': model.encoder.geomgcn1.parameters(), 'weight_decay': args.weight_decay_layer_one},
            {'params': model.decoder.parameters(), 'weight_decay': args.weight_decay_mlp}
        ],
        lr=args.geomlr
    )
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={param_group['lr']}, weight_decay={param_group['weight_decay']}")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True)

    epoch = []
    loss = []
    acc = []
    auc_roc = []
    auprc = []
    val_loss = []
    val_acc = []
    val_auc_roc = []
    val_auprc = []

    profiler = None
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as p:
        for ep in range(1, NUM_EPOCHS + 1):
            if ep <= 10:
                train_loss, train_accuracy, train_ROCAUC, train_AUPRC, valid_loss, val_accuracy, val_ROCAUC, val_AUPC = train_with_autocast(
                    model, ep, scaler=scaler, profiler=p)
            else:
                train_loss, train_accuracy, train_ROCAUC, train_AUPRC, valid_loss, val_accuracy, val_ROCAUC, val_AUPC = train_with_autocast(model, ep, scaler=scaler)

            print(
                'Train epoch: {} - loss: {:.4f} - accuracy: {:.4f} - ROCAUC: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f} - val_ROCAUC: {:.4f}'.format(
                    ep, train_loss.item(), train_accuracy, train_ROCAUC, valid_loss, val_accuracy,
                    val_ROCAUC))
            torch.cuda.empty_cache()

            if not os.path.exists(root_model):
                os.makedirs(root_model)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            torch.save(model.state_dict(),
                   os.path.join(save_model_path, 'TridentTCR_epoch{:03d}_AUC{:.6f}.pth'.format(ep, val_ROCAUC.item())))
            loss.append(train_loss.item())
            acc.append(train_accuracy.item())
            auc_roc.append(train_ROCAUC.item())
            auprc.append(train_AUPRC.item())
            val_loss.append(valid_loss.item())
            val_acc.append(val_accuracy.item())
            val_auc_roc.append(val_ROCAUC.item())
            val_auprc.append(val_AUPC.item())
            epoch.append(ep)
            scheduler.step(valid_loss)

            if val_ROCAUC > best_auc:
                best_auc = val_ROCAUC
                best_auc_epoch = ep
                torch.save(model.state_dict(), best_auc_model_path)

        print(f"Best AUC: {best_auc:.4f} at Epoch {best_auc_epoch}")

        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Logs of results
    dfhistory = {'epoch': epoch,
                 'loss': loss, 'acc': acc, 'auc_roc': auc_roc, 'auprc': auprc,
                 'val_loss': val_loss, 'val_acc': val_acc, 'val_auc_roc': val_auc_roc, 'val_auprc': val_auprc}
    df = pd.DataFrame(dfhistory)
    if not os.path.exists(root_history):
        os.makedirs(root_history)
    df.to_csv(os.path.join(root_history, "TridentTCR_{}.tsv".format(model_dir)), header=True, sep='\t', index=False)

    # get num_epoch
    val_auc_roc = list(df['val_auc_roc'])
    max_auc_index = val_auc_roc.index(max(val_auc_roc))
    num_epoch = max_auc_index + 1
    print("max auc epoch:", num_epoch)
    print(max(val_auc_roc))
    val_acc = list(df['val_acc'])
    print("ACC epoch:", num_epoch)
    print(val_acc[max_auc_index])

    val_loss = list(df['val_loss'])
    min_val_loss_index = val_loss.index(min(val_loss))
    loss_num_epoch = min_val_loss_index + 1
    print("loss auc epoch:", loss_num_epoch)
    print(val_auc_roc[min_val_loss_index])

    model.load_state_dict(torch.load(best_auc_model_path))

    with torch.no_grad():
        with autocast():
            out_test,_ = model(g_test, g_test.ndata['feat'], test_edge_label_index)
            if args.mode == 'binary':
                test_predictions = torch.sigmoid(out_test)
                tp, tn, fp, fn, _, _, _, _ = utils_functions.calculate_binary_confusion_matrix(test_predictions, y_test)
            else:
                test_predictions = F.softmax(out_test, dim=1)
                tp, tn, fp, fn, _, _, _, _  = utils_functions.calculate_trinary_confusion_matrix(test_predictions, y_test, num_classes=3)
            print(f"Max AUC Epoch {best_auc_epoch}: TP={tp}, TN={tn},FP={fp}, FN={fn}")
    # Remove useless models
    for root, dirs, files in os.walk(save_model_path):
        for file in files:
            if file.startswith("TridentTCR_epoch{:03d}".format(loss_num_epoch)):
                os.rename(os.path.join(save_model_path, file), os.path.join(save_model_path, 'minloss_AUC_' + file))
            elif file.startswith("TridentTCR_epoch{:03d}".format(num_epoch)):
                os.rename(os.path.join(save_model_path, file), os.path.join(save_model_path, 'max_AUC_' + file))
            else:
                os.remove(os.path.join(save_model_path, file))


