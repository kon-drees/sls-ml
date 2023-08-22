import os

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, BatchNorm1d
from torch.nn import functional as F

from torch_geometric.data import DataLoader, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from joblib import Parallel, delayed
from tqdm import tqdm

from sls_ml.af_parser import parse_file


class AAF_GCNConv(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(AAF_GCNConv, self).__init__()
        self.pre_layer = Linear(num_features, 256)  # Pre-message passing layer
        self.batch_norm1 = BatchNorm1d(256)
        self.conv1 = GCNConv(256, 256)  # First message passing layer
        self.batch_norm2 = BatchNorm1d(256)
        self.conv2 = GCNConv(256, 256)  # Second message passing layer
        self.batch_norm3 = BatchNorm1d(256)
        self.conv3 = GCNConv(256, 256)  # Third message passing layer
        self.batch_norm4 = BatchNorm1d(256)
        self.post_layer1 = Linear(256, 256)  # First post-message passing layer
        self.batch_norm5 = BatchNorm1d(256)
        self.post_layer2 = Linear(256, num_classes)  # Second post-message passing layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.pre_layer(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.post_layer1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.post_layer2(x)

        return F.log_softmax(x, dim=1)


class AAF_GraphSAGE_Conv(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(AAF_GraphSAGE_Conv, self).__init__()
        self.pre_layer = torch.nn.Linear(num_features, 256)
        self.batch_norm1 = torch.nn.BatchNorm1d(256)
        self.conv1 = SAGEConv(256, 256)
        self.batch_norm2 = torch.nn.BatchNorm1d(256)
        self.conv2 = SAGEConv(256, 256)
        self.batch_norm3 = torch.nn.BatchNorm1d(256)
        self.conv3 = SAGEConv(256, 256)
        self.batch_norm4 = torch.nn.BatchNorm1d(256)
        self.post_layer1 = torch.nn.Linear(256, 256)
        self.batch_norm5 = torch.nn.BatchNorm1d(256)
        self.post_layer2 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.pre_layer(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.post_layer1(x)
        x = self.batch_norm5(x)
        x = F.relu(x)
        x = self.post_layer2(x)

        return F.log_softmax(x, dim=1)


def load_and_preprocess_data_random_for_dataloader(processed_feature_file, argumentation_files, processed_feature_folder, processed_label_folder, argumentation_folder, reduced):
    graph_name = os.path.splitext(processed_feature_file)[0]
    if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
        return None
    processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
    x_train_data = pd.read_csv(processed_feature_file_path)
    if (reduced):
        x_train_data.drop(
            ['in_degree_centrality', 'predecessors_neighbors', 'degree_centrality', 'average_neighbor_degree',
             'out_degree_centrality', 'successors_neighbors', 'edge_node_ratio', 'successors'], axis=1, inplace=True)

    processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
    y_train_data = pd.read_csv(processed_label_file_path)

    x_train_data_dict = x_train_data.set_index('argument').to_dict('index')
    y_train_data_dict = y_train_data.set_index('Argument').to_dict('index')

    graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
    if not os.path.exists(graph_file):
        graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')
    argumentation_framework = parse_file(graph_file)


    data = from_networkx(argumentation_framework)
    features = []
    labels = []

    for arg in x_train_data['argument']:

        feature_row = x_train_data_dict[arg].copy()
        label_row = y_train_data_dict[arg]

        in_stability_impact = label_row['In_Stability_Impact']
        out_stability_impact = label_row['Out_Stability_Impact']

        feature_row_in = feature_row.copy()
        feature_row_in['state'] = 1  # '1' for 'in' state

        feature_row_in_values = pd.DataFrame([feature_row_in]).values.flatten()
        features.append(feature_row_in_values)

        labels.append(in_stability_impact)

        feature_row_out = feature_row.copy()
        feature_row_out['state'] = 0  # '0' for 'out' state

        feature_row_out_values = pd.DataFrame([feature_row_out]).values.flatten()
        features.append(feature_row_out_values)

        labels.append(out_stability_impact)

    features_np = np.array(features)
    labels_np = np.array(labels)
    data.x = torch.tensor(features_np, dtype=torch.float)
    data.y = torch.tensor(labels_np, dtype=torch.long)

    return data


def create_dataloader_random(argumentation_folder, processed_feature_folder, processed_label_folder, reduced):
    processed_feature_files = os.listdir(processed_feature_folder)
    argumentation_files = os.listdir(argumentation_folder)

    data_list = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_random_for_dataloader)(processed_feature_file, argumentation_files, processed_feature_folder,
                                                 processed_label_folder, argumentation_folder, reduced)
        for processed_feature_file in tqdm(processed_feature_files)
    )

    data_list = [data for data in data_list if data is not None]  # Filter out None values
    loader = DataLoader(data_list, batch_size=32, shuffle=True, collate_fn=Batch.from_data_list)

    return loader


def train_model_random(model, argumentation_folder, processed_feature_folder, processed_label_folder, reduced = False):

    loader = create_dataloader_random(argumentation_folder, processed_feature_folder, processed_label_folder, reduced)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(500):
        print('EPOCH {}:'.format(epoch + 1))
        for data in loader:
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_preds = []
        epoch_labels = []
        for data in loader:
            predictions = model(data).max(dim=1)[1]
            correct = predictions.eq(data.y).sum().item()
            acc = correct / len(data.y)
            print('Epoch: {:03d}, Accuracy: {:.4f}'.format(epoch + 1, acc))

            epoch_preds.extend(predictions.tolist())
            epoch_labels.extend(data.y.tolist())

    accuracy = acc
    roc_auc = roc_auc_score(epoch_labels, epoch_preds)
    report = classification_report(epoch_labels, epoch_preds)
    matrix = confusion_matrix(epoch_labels, epoch_preds)
    class_balance_ratio = len([i for i in epoch_labels if i == 0]) / len([i for i in epoch_labels if i == 1])
    class_distribution = {0: epoch_labels.count(0), 1: epoch_labels.count(1)}

    # Save the metrics
    with open('../files/metrics/metrics_rn_red.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'ROC AUC: {roc_auc}\n')
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Confusion Matrix:\n{matrix}\n')
        f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
        f.write(f'Class Distribution: {class_distribution}\n')

    return model


def load_and_preprocess_data_initial_for_dataloader(processed_feature_file, argumentation_files, processed_feature_folder, processed_label_folder, argumentation_folder, reduced):
    graph_name = os.path.splitext(processed_feature_file)[0]
    if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
        return None


    processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
    x_train_data = pd.read_csv(processed_feature_file_path)

    if(reduced):
        x_train_data.drop(['in_degree_centrality', 'predecessors_neighbors', 'degree_centrality','average_neighbor_degree','out_degree_centrality', 'successors_neighbors', 'edge_node_ratio', 'successors' ], axis=1, inplace=True)

    processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
    y_train_data = pd.read_csv(processed_label_file_path)

    x_train_data_dict = x_train_data.set_index('argument').to_dict('index')
    y_train_data_dict = y_train_data.set_index('Argument').to_dict('index')

    graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
    if not os.path.exists(graph_file):
        graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')
    argumentation_framework = parse_file(graph_file)

    data = from_networkx(argumentation_framework)

    features = []
    labels = []

    for arg in x_train_data['argument']:

        feature_row = x_train_data_dict[arg].copy()
        label_row = y_train_data_dict[arg]

        label = label_row['Label']
        label = 1 if label == 'in' else 0  # Convert 'in' to 1 and 'out' to 0

        feature_row_values = pd.DataFrame([feature_row]).values.flatten()
        features.append(feature_row_values)

        labels.append(label)

    features_np = np.array(features)
    labels_np = np.array(labels)
    data.x = torch.tensor(features_np, dtype=torch.float)
    data.y = torch.tensor(labels_np, dtype=torch.long)

    return data


def create_dataloader_initial(argumentation_folder, processed_feature_folder, processed_label_folder, reduced):
    processed_feature_files = os.listdir(processed_feature_folder)
    argumentation_files = os.listdir(argumentation_folder)

    data_list = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_initial_for_dataloader)(processed_feature_file, argumentation_files,
                                                                processed_feature_folder,
                                                                processed_label_folder, argumentation_folder, reduced)
        for processed_feature_file in tqdm(processed_feature_files)
    )

    data_list = [data for data in data_list if data is not None]  # Filter out None values
    loader = DataLoader(data_list, batch_size=32, shuffle=True, collate_fn=Batch.from_data_list)
    return loader


def train_model_inital(model, argumentation_folder, processed_feature_folder, processed_label_folder, reduced = False):

    loader = create_dataloader_initial(argumentation_folder, processed_feature_folder, processed_label_folder, reduced)

    for batch in loader:
        features = batch.x
        labels = batch.y
        print('Loaded features shape:', features.shape)  # Debug print statement
        print('Loaded labels shape:', labels.shape) # Debug print statement

    num_features = features.shape[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"Number of features: {num_features}")

    # Training loop
    for epoch in range(500):
        print('EPOCH {}:'.format(epoch + 1))
        for data in loader:
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_preds = []
        epoch_labels = []
        for data in loader:
            predictions = model(data).max(dim=1)[1]
            correct = predictions.eq(data.y).sum().item()
            acc = correct / len(data.y)
            print('Epoch: {:03d}, Accuracy: {:.4f}'.format(epoch + 1, acc))

            epoch_preds.extend(predictions.tolist())
            epoch_labels.extend(data.y.tolist())

    accuracy = acc
    roc_auc = roc_auc_score(epoch_labels, epoch_preds)
    report = classification_report(epoch_labels, epoch_preds)
    matrix = confusion_matrix(epoch_labels, epoch_preds)
    class_balance_ratio = len([i for i in epoch_labels if i == 0]) / len([i for i in epoch_labels if i == 1])
    class_distribution = {0: epoch_labels.count(0), 1: epoch_labels.count(1)}
        # Save the metrics
    with open('../files/metrics/metrics_in_red.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Confusion Matrix:\n{matrix}\n')
            f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
            f.write(f'Class Distribution: {class_distribution}\n')

    return model


if __name__ == '__main__':
    # Paths
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_feature_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'

    processed_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    processed_label_folder_in = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_initial_argumentation_frameworks'

    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    model_in = AAF_GraphSAGE_Conv(12, 2)
    model_in = train_model_inital(model_in, argumentation_folder, processed_feature_folder, processed_label_folder_in)
    PATH = os.path.join(output_folder, "g_nn_in.pt")
    torch.save(model_in.state_dict(), PATH)

    model_rn = AAF_GraphSAGE_Conv(13, 2)
    model_rn = train_model_random(model_rn, argumentation_folder, processed_feature_folder, processed_label_folder)
    PATH = os.path.join(output_folder, "g_nn_rn.pt")
    torch.save(model_rn.state_dict(), PATH)

    model_in_red = AAF_GCNConv(4, 2)
    model_in_red = train_model_inital(model_in_red, argumentation_folder, processed_feature_folder, processed_label_folder_in, True)
    PATH = os.path.join(output_folder, "g_nn_in_red.pt")
    torch.save(model_in_red.state_dict(), PATH)

    model_rn_red = AAF_GCNConv(5, 2)
    model_rn_red = train_model_random(model_rn_red, argumentation_folder, processed_feature_folder, processed_label_folder, True)
    PATH = os.path.join(output_folder, "g_nn_rn_red.pt")
    torch.save(model_rn_red.state_dict(), PATH)

