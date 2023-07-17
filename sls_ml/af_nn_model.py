import torch
from torch.nn import Linear
from torch.nn import functional as F
from torch_geometric import loader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm

# Define the GCN model
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.pre_layer = Linear(num_features, 256)  # Pre-message passing layer
        self.batch_norm1 = BatchNorm(256)
        self.conv1 = GCNConv(256, 256)  # First message passing layer
        self.batch_norm2 = BatchNorm(256)
        self.conv2 = GCNConv(256, 256)  # Second message passing layer
        self.batch_norm3 = BatchNorm(256)
        self.conv3 = GCNConv(256, 256)  # Third message passing layer
        self.batch_norm4 = BatchNorm(256)
        self.post_layer1 = Linear(256, 256)  # First post-message passing layer
        self.batch_norm5 = BatchNorm(256)
        self.post_layer2 = Linear(256, num_classes)  # Second post-message passing layer

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

# Training settings
model = Net(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(1000):
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
