import torch


def update_features_info(previous_sum, previous_counts, features, labels):
    features = features.detach().cpu()
    labels = labels.detach().cpu()
    class_sums = torch.zeros_like(previous_sum)
    class_batch_counts = torch.zeros_like(previous_counts, dtype=torch.long)

    for feature, label in zip(features, labels):
        class_sums[label] += feature
        class_batch_counts[label] += 1

    previous_sum += class_sums
    previous_counts += class_batch_counts

    # for i in range(num_classes):
    #     if class_batch_counts[i] > 0:
    #         total_count = previous_counts[i] + class_batch_counts[i]
    #
    #         previous_sum[i] += class_sums[i]
    #         previous_counts[i] = total_count

    return previous_sum, previous_counts


def calculate_class_averages(class_sums, class_counts):
    class_averages = class_sums / class_counts.view(-1, 1)

    return class_averages
