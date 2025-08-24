from Process.Valid import valid


def evaluate_all_models(args, models, target_loder, device):
    logger = args.logger

    for i, model in enumerate(models):
        logger.info(f"Running evaluation on model {i}.")
        model.to(device)
        accuracy, cacc = valid(args, model, target_loder, device)

        logger.info(f"Accuracy on model {i}: {accuracy}")


def evaluate_single_model(args, model, target_loder, device):
    logger = args.logger

    model.to(device)
    accuracy, cacc = valid(args, model, target_loder, device)

    logger.info(f"Accuracy on model: {accuracy}")

    return accuracy, cacc
