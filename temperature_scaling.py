# script from https://github.com/gpleiss/temperature_scaling with modifications

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.functional.classification import multiclass_calibration_error

class ModelWithTemperature(nn.Module):
    """
    code from https://github.com/gpleiss/temperature_scaling with modifications
    
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.num_classes = model.num_classes

    def forward(self, input):
        encoded = self.model.encoder(input)
        decoded = self.model.decoder(encoded)
        logits = self.model.classifier(encoded)
        probs = F.softmax(self.temperature_scale(logits), dim=1)
        return encoded, decoded, probs

    def forward_logits(self, input):
        encoded = self.model.encoder(input)
        decoded = self.model.decoder(encoded)
        logits = self.model.classifier(encoded)
        logits = self.temperature_scale(logits)
        return encoded, decoded, logits

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

class PooledModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(PooledModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.num_classes = model.num_classes

    def forward(self, input):
        input = self.model.encoder(input)
        input, indices = self.model.maxpool(input)
        encoded = self.model.fc_encode(input)
        logits = self.model.classifier(encoded)
        probs = F.softmax(self.temperature_scale(logits), dim=1)
        decoded = self.model.fc_decode(encoded)
        decoded = self.model.maxunpool(decoded, indices)
        decoded = self.model.decoder(decoded)
        return encoded, decoded, probs

    def forward_logits(self, input):
        input = self.model.encoder(input)
        input, indices = self.model.maxpool(input)
        encoded = self.model.fc_encode(input)
        logits = self.model.classifier(encoded)
        logits = self.temperature_scale(logits)
        decoded = self.model.fc_decode(encoded)
        decoded = self.model.maxunpool(decoded, indices)
        decoded = self.model.decoder(decoded)
        return encoded, decoded, logits

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def set_temperature(modelwtemp, valid_loader):
    """
    Tune the temperature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    nll_criterion = nn.CrossEntropyLoss()

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            _, _, logits = modelwtemp.forward_logits(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = multiclass_calibration_error(logits, labels, num_classes=10, n_bins=15, norm='l1')
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([modelwtemp.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(modelwtemp.temperature_scale(logits), labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(modelwtemp.temperature_scale(logits), labels).item()
    after_temperature_ece = multiclass_calibration_error(modelwtemp.temperature_scale(logits), labels, num_classes=10, n_bins=15, norm='l1')
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
    
    print('Optimal temperature: %.3f' % modelwtemp.temperature.item())