import os

import pandas as pd
import torch
from torch.nn import Linear, BatchNorm1d
from torch.nn import functional as F

from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

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


def create_dataloader_random(argumentation_folder, processed_feature_folder, processed_label_folder):
    processed_feature_files = os.listdir(processed_feature_folder)
    argumentation_files = os.listdir(argumentation_folder)

    # Initialize lists to store Data objects and labels
    data_list = []

    for processed_feature_file in processed_feature_files:
        graph_name = os.path.splitext(processed_feature_file)[0]
        if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
            continue

        # Load the processed feature data from CSV
        processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
        x_train_data = pd.read_csv(processed_feature_file_path)

        # Load the processed label data from CSV
        processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
        y_train_data = pd.read_csv(processed_label_file_path)

        # Load the graph
        graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
        if not os.path.exists(graph_file):
            graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')
        argumentation_framework = parse_file(graph_file)
        edge_node_ratio = argumentation_framework.number_of_edges() / argumentation_framework.number_of_nodes()

        # Convert NetworkX graph to PyTorch Geometric Data
        data = from_networkx(argumentation_framework)

        # Prepare the features and labels
        features = []
        labels = []
        for arg in x_train_data['argument']:
            feature_row = x_train_data[x_train_data['argument'] == arg].drop('argument', axis=1)
            label_row = y_train_data[y_train_data['Argument'] == arg]

            in_stability_impact = label_row['In_Stability_Impact'].values[0]
            out_stability_impact = label_row['Out_Stability_Impact'].values[0]

            # Add edge_node_ratio to the feature row
            feature_row['edge_node_ratio'] = edge_node_ratio

            # Create two identical rows for the same argument, one for 'in' and another for 'out'
            feature_row_in = feature_row.copy()
            feature_row_in['state'] = 1  # '1' for 'in' state
            feature_row_in_values = feature_row_in.values.flatten()
            features.append(feature_row_in_values)

            labels.append(in_stability_impact)

            feature_row_out = feature_row.copy()
            feature_row_out['state'] = 0  # '0' for 'out' state
            feature_row_out_values = feature_row_out.values.flatten()
            features.append(feature_row_out_values)

            labels.append(out_stability_impact)

        # Convert features and labels to tensors
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels, dtype=torch.long)

        data_list.append(data)

    loader = DataLoader(data_list, batch_size=32, shuffle=True, collate_fn=Batch.from_data_list)

    return loader



def train_model_random(argumentation_folder, processed_feature_folder, processed_label_folder):
    # Training settings
    loader = create_dataloader_random(argumentation_folder, processed_feature_folder, processed_label_folder)

    # load the first batch
    dataiter = iter(loader)
    for batch in loader:
        features = batch.x  # Access the node features
        labels = batch.y  # Access the labels
        print('Loaded features shape:', features.shape)  # Debug print statement
        print('Loaded labels shape:', labels.shape)

    # get the number of features
    num_features = features.shape[1]
    model = AAF_GCNConv(13, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"Number of features: {num_features}")

    # Training loop
    for epoch in range(400):
        print('EPOCH {}:'.format(epoch + 1))
        for data in loader:
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        for data in loader:
            predictions = model(data).max(dim=1)[1]
            correct = predictions.eq(data.y).sum().item()
            acc = correct / len(data.y)
            print('Epoch: {:03d}, Accuracy: {:.4f}'.format(epoch, acc))

    return model


def create_dataloader_initial(argumentation_folder, processed_feature_folder, processed_label_folder):
    processed_feature_files = os.listdir(processed_feature_folder)
    argumentation_files = os.listdir(argumentation_folder)

    # Initialize lists to store Data objects and labels
    data_list = []

    for processed_feature_file in processed_feature_files:
        graph_name = os.path.splitext(processed_feature_file)[0]
        if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
            continue

        # Load the processed feature data from CSV
        processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
        x_train_data = pd.read_csv(processed_feature_file_path)

        # Load the processed label data from CSV
        processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
        y_train_data = pd.read_csv(processed_label_file_path)

        # Load the graph
        graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
        if not os.path.exists(graph_file):
            graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')
        argumentation_framework = parse_file(graph_file)
        edge_node_ratio = argumentation_framework.number_of_edges() / argumentation_framework.number_of_nodes()

        # Convert NetworkX graph to PyTorch Geometric Data
        data = from_networkx(argumentation_framework)

        # Prepare the features and labels
        features = []
        labels = []
        for arg in x_train_data['argument']:
            feature_row = x_train_data[x_train_data['argument'] == arg].drop('argument', axis=1)
            label_row = y_train_data[y_train_data['Argument'] == arg]

            in_stability_impact = label_row['In_Stability_Impact'].values[0]
            out_stability_impact = label_row['Out_Stability_Impact'].values[0]

            # Add edge_node_ratio to the feature row
            feature_row['edge_node_ratio'] = edge_node_ratio

            # Create two identical rows for the same argument, one for 'in' and another for 'out'
            feature_row_in = feature_row.copy()
            feature_row_in['state'] = 1  # '1' for 'in' state
            feature_row_in_values = feature_row_in.values.flatten()
            features.append(feature_row_in_values)

            labels.append(in_stability_impact)

            feature_row_out = feature_row.copy()
            feature_row_out['state'] = 0  # '0' for 'out' state
            feature_row_out_values = feature_row_out.values.flatten()
            features.append(feature_row_out_values)

            labels.append(out_stability_impact)

        # Convert features and labels to tensors
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels, dtype=torch.long)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=32, shuffle=True, collate_fn=Batch.from_data_list)
    return loader

if __name__ == '__main__':
    # Paths
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_feature_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'
    processed_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    model = train_model_random(argumentation_folder,processed_feature_folder,processed_label_folder)

