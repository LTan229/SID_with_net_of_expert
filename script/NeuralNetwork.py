import torch.nn as nn

class Args():
    input_dim = 400    # the number of input dimensions
    output_dim = 250        # the number of outputs (i.e., # classes on MNIST)
    lamda = 1e-1           # coefficient of the regularization term
    lr = 1e-3              # learning rate
    weight_decay = 1e-5   # weight decay
    batch_size = 128       # batch size
    epochs = 50            # the number of training epochs
    log_interval = 100     # the number of batches to wait before printing logs
    use_cuda = False       # whether to use GPU
    test_batch_size = 1000
    name = 'temp'
    seed = 0
    num_classes = 250

class NeuralNetwork(nn.Module):
    def __init__(self, out_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, out_dim),
        )
        self.training_loss_list = []
        self.testing_acc_list = []
        self.best_testing_acc = 0

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits