import os
from Process.Metrics import AverageMeter
from Process.Valid import valid
from Utils.utils import save_model
from loss.kl_loss import SmoothedKLDivergenceLoss
from Utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from Utils.utils import update_details_file

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm

logger = None


def train_model(args, student_model, source_models, target_loader, round_id, valida_data_loader):
    global logger
    logger = args.logger

    logger.info(f"Start training main model using {len(source_models)} source models.")

    log_dir = os.path.join(args.output_dir, "logs", str(round_id), "main_model")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    optimizer = torch.optim.SGD(student_model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.main_steps * len(target_loader)
    warmup_steps = t_total * 0.05

    logger.info(f"Warmup steps for main model: {warmup_steps}")

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    student_model.to(args.device)
    student_model.train()
    student_model.zero_grad()
    losses = AverageMeter()

    global_step, best_acc = 0, 0

    loss_slce = SmoothedKLDivergenceLoss(num_classes=args.num_classes, epsilon=0.5)

    target_loader = torch.utils.data.DataLoader(
        target_loader.dataset,
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    for i in range(args.main_steps):
        logger.info(f"Training main model, epoch {i}.")

        target_loader_iter = tqdm(target_loader,
                                  desc=f"Training main model (X / X Steps), Epoch X, (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

        for step, batch in enumerate(target_loader_iter):
            x_source, y_source, pseudo_labels, indices = batch
            x_source = x_source.to(args.device)

            pseudo_labels = []
            for source_model in source_models:
                source_model = source_model.to(args.device)
                source_model.eval()
                with torch.no_grad():
                    _, logits, features_backbone_source = source_model(x_source, return_features_backbone=True)
                    logits = nn.functional.softmax(logits, dim=1)
                    pseudo_labels.append(logits)

            pseudo_labels = torch.stack(pseudo_labels, dim=0)
            pseudo_labels = torch.mean(pseudo_labels, dim=0)

            student_model.train()

            source_features, logits, features_backbone = student_model(x_source, return_features_backbone=True)
            logits = nn.functional.softmax(logits, dim=1)

            loss = loss_slce(logits, pseudo_labels)

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            target_loader_iter.set_description(
                f"Training main model (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

        logger.info("Evaluating trained model.")
        accuracy, cacc = valid(args, student_model, valida_data_loader, args.device)
        logger.info(f"New model accuracy is: {accuracy}")

        if accuracy > best_acc:
            best_acc = accuracy
            logger.info(f"Accuracy improved. New best accuracy is: {best_acc}")
            save_model(args, student_model, model_name=f"main_model_{str(round_id)}_best_{args.name}")

        logger.warning(f"Main Model, loss for epoch {i}: {losses.avg}")

        eta = 0.1
        student_state_dict = student_model.state_dict()

        for source_model in source_models:
            source_state_dict = source_model.state_dict()
            for key in source_state_dict:
                source_state_dict[key] = (1 - eta) * source_state_dict[key] + eta * student_state_dict[key]

            source_model.load_state_dict(source_state_dict)

        writer.add_scalar('Loss/train', losses.avg, i)
        losses.reset()

    if valida_data_loader:
        accuracy, cacc = valid(args, student_model, valida_data_loader, args.device)
        update_details_file(args, f"Accuracy of the model after training with pseudo labels: {accuracy}")
        logger.warning(f"Best Main model, Validation accuracy: {best_acc}")
        if cacc is not None:
            writer.add_scalar('ClassAccuracy/valid', cacc)
        writer.add_scalar('Accuracy/valid', accuracy)

    save_model(args, student_model, model_name=f"main_model_{str(round_id)}_{args.name}")
    writer.close()
    return student_model, accuracy