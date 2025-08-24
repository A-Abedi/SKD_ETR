import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn


def calculate_entropy_for_model(model, target_loader, devices, multi = True):
    device = f"cuda:{devices[0]}"

    total_entropy = 0.0
    total_samples = 0

    if len(devices) > 1 and multi:
        model = nn.DataParallel(model, device_ids=devices)

    epoch_iterator = tqdm(target_loader, desc="Calculate Entropy for model", leave=False)

    model.eval()
    model.to(device)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            x, y, _, _ = batch
            x = x.to(device)
            outputs = model(x, return_features_only=True)

            probabilities = F.softmax(outputs, dim=1)

            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)

            total_entropy += torch.sum(entropy).item()
            total_samples += x.size(0)

    avg_entropy = total_entropy / total_samples
    return avg_entropy


def transform_and_scale_weights(weights, alpha=2, beta=2):
    weights = np.array(weights)

    mu = np.mean(weights)

    transformed_weights = np.zeros_like(weights)

    for i, w in enumerate(weights):
        ratio = w / mu
        scale = alpha if w > mu else beta
        transformed_weights[i] = ratio ** scale

    transformed_weights /= np.sum(transformed_weights)

    return transformed_weights.round(2)


def calculate_weights_based_on_entropy(models, target_loader, devices):
    entropies = []

    for model in models:
        entropy = calculate_entropy_for_model(model, target_loader, devices)
        entropies.append(entropy)

    entropies = torch.tensor(entropies)

    weights = 1.0 / (entropies + 1e-8)

    weights = weights / torch.sum(weights)

    final_weights = transform_and_scale_weights(weights.tolist())
    return final_weights.tolist()
