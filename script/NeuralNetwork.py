import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, out_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(400, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Softmax()
        )
        self.training_loss_list = []
        self.testing_acc_list = []
        self.best_testing_acc = 0

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits