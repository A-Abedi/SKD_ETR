import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from tqdm import tqdm

from Utils.transform import get_transform
from Utils.data_utils import ImageList
from Utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from Utils.utils import save_model, update_details_file
from Model import get_backbone, get_bottleneck, get_head, FUDA
from .Metrics import AverageMeter
from .Valid import valid
from Federated.TrainMainModel import train_model


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

logger = None
mp.set_start_method('spawn', force=True)


def get_model(args, client_id, round_id, return_path: bool = True):
    model_name = str(client_id) + f"_{str(round_id)}" + "_last"
    if not return_path:
        return model_name

    model_path = os.path.join(args.output_dir, "%s.bin" % model_name)

    return model_path


def train_one_client(args, data_loader, model: nn.Module, client_id, round_id, device, valida_data_loader=None):
    log_dir = os.path.join(args.output_dir, "logs", str(round_id), client_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.num_steps_clients * len(data_loader)
    warmup_steps = t_total * 0.05

    logger.warning(f"Warmup steps, Device: {device}, Client: {client_id} is {warmup_steps}")

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    model.train()
    model.zero_grad()
    losses = AverageMeter()
    global_step, bes_loss = 0, 1e10
    loss_fct = torch.nn.CrossEntropyLoss()

    # source_features_path = os.path.join(args.output_dir, f"FeaturesMean_{client_id}.pt")
    #
    # previous_sum = torch.zeros(args.num_classes, model.backbone.norm.normalized_shape[0])
    # previous_counts = torch.zeros(args.num_classes, dtype=torch.long)

    for i in range(args.num_steps_clients):
        epoch_iterator = tqdm(data_loader,
                              desc=f"Device: {device}, Client: {client_id}, Training (X / X Steps), Epoch X, (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              position=device)

        for step, batch in enumerate(epoch_iterator):
            x_source, y_source, _, _ = batch
            x_source = x_source.to(device)
            y_source = y_source.to(device)

            source_features, logits, features_backbone = model(x_source, return_features_backbone=True)

            # previous_sum, previous_counts = update_features_info(previous_sum,
            #                                                      previous_counts,
            #                                                      features_backbone,
            #                                                      y_source)

            loss = loss_fct(logits.view(-1, model.head.out_features), y_source.view(-1))

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                f"Device: {device}, Client: {client_id}, Training (%d / %d Steps), Epoch %d, (loss=%2.5f)" %
                (global_step, t_total, i, losses.val)
            )

        # class_averages = calculate_class_averages(previous_sum, previous_counts)
        # torch.save(class_averages, source_features_path)

        writer.add_scalar('Loss/train', losses.avg, global_step)
        losses.reset()

    if valida_data_loader:
        accuracy, cacc = valid(args, model, valida_data_loader, device)
        update_details_file(args, f"Device {device}, Client: {client_id} accuracy: {accuracy}")
        logger.warning(f"device: {device}, Validation accuracy: {accuracy}")
        if cacc is not None:
            writer.add_scalar('ClassAccuracy/valid', cacc)
        writer.add_scalar('Accuracy/valid', accuracy)

    writer.close()
    save_model(args, model, model_name=str(client_id) + f"_{str(round_id)}" + "_last")
    return model


def train_one_client_on_gpu(rank, args, source_loader, model, client_id, round_id, world_size, valid_loader=None):
    global logger
    logger = args.logger

    torch.cuda.set_device(args.gpu_ids[rank])

    model = model.to(args.gpu_ids[rank])

    train_one_client(args, source_loader, model, client_id, round_id, args.gpu_ids[rank], valid_loader)


def save_config_to_file(args, filename="Details.txt"):
    questions = {"Round? ": args.num_rounds,
                 "Steps per client? ": args.num_steps_clients,
                 "Sources? ": args.source_list,
                 "Target? ": args.target_list,
                 "Num Classes? ": args.num_classes,
                 "Backbone? ": args.backbone_size,
                 "Bottleneck? ": args.bottleneck_size,
                 "Additional info? ": "Nothing"}

    additional_info = input("Additional info ? ")

    questions["Additional info? "] = additional_info

    content = "\n".join([f"{question.strip()} {answer}" for question, answer in questions.items()])

    file_path = os.path.join(args.output_dir, filename)

    with open(file_path, "a") as file:
        file.write(content)


def train(args):
    global logger
    logger = args.logger
    """ Train the model """
    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)

    args.output_dir = os.path.join(args.output_dir, args.dataset, f"Experiment {args.exp_num}")

    args.detail_file_name = "Details.txt"

    if args.train_from_scratch:
        os.makedirs(args.output_dir)
        save_config_to_file(args)
    else:
        update_details_file(args, "########## Train from previous trained clients ##########")
        save_config_to_file(args)

    args.train_batch_size = args.train_batch_size
    transform_train, transform_test = get_transform(args.img_size)

    source_loaders = []
    source_ids = []
    for source_list in args.source_list:
        source_list_lines = open(source_list).readlines()
        source_loader = torch.utils.data.DataLoader(
            ImageList(source_list_lines, args.dataset_path, transform=transform_train, mode='RGB'),
            batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=False
        )

        source_loaders.append(source_loader)
        source_id = source_list.split('/')[-1].split(".")[0]

        source_ids.append(source_id)

    target_data_list = open(args.target_list).readlines()

    target_loader = torch.utils.data.DataLoader(
        ImageList(target_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        ImageList(target_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs for each client in each round = %d", args.num_steps_clients)
    logger.info("  Number of rounds = %d", args.num_rounds)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    backbone = get_backbone(args.backbone_type, args.backbone_size)

    for name, param in backbone.named_parameters():
        param.requires_grad = False

    out_feature_size = backbone.norm.normalized_shape[0]

    num_gpus = len(args.gpu_ids)

    num_processes = min(num_gpus, len(source_loaders))

    logger.info(f"Number of processes: {num_processes}")

    init_model = False

    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    best_acc = 0
    for round in range(args.num_rounds):
        models = []
        logger.info(f"Starting round {round}.")
        if args.train_from_scratch:
            for i in range(0, len(source_loaders), num_processes):
                current_loaders = source_loaders[i:i + num_processes]
                current_ids = source_ids[i:i + num_processes]
                processes = []

                for j, (source_id, source_loader) in enumerate(zip(current_ids, current_loaders)):
                    logger.info(f"\nStarting training for source id {source_id}.")
                    bottleneck = get_bottleneck(args.bottleneck_size, out_feature_size)
                    head = get_head(args.num_classes)
                    model = FUDA(backbone, bottleneck, head)

                    if init_model:
                        logger.info(f"Initiated the model {source_id} using previous trained model.")
                        model.load_state_dict(init_model)

                    p = mp.Process(target=train_one_client_on_gpu, args=(j, args, source_loader, model, source_id, round,
                                                                         num_processes, test_loader))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

        for source_id in source_ids:
            model_name = get_model(args, source_id, round)
            logger.info(f"Loading model {model_name}")
            model_state_dict = torch.load(model_name, map_location="cpu")
            model = FUDA(backbone, get_bottleneck(args.bottleneck_size, out_feature_size), get_head(args.num_classes))
            model.load_state_dict(model_state_dict)
            model.eval()
            models.append(model)

        args.device = f"cuda:{args.gpu_ids[0]}"

        student_model = FUDA(backbone,
             get_bottleneck(args.bottleneck_size, out_feature_size),
             get_head(args.num_classes))

        main_model, accuracy = train_model(args, student_model, models, target_loader, round, test_loader)
        main_model.to("cpu")
        init_model = main_model.state_dict()

        new_state_dict = {}
        for key in init_model.keys():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = init_model[key]

        init_model = new_state_dict

        writer.add_scalar('Accuracy/valid', accuracy, round)

        if accuracy > best_acc:
            best_acc = accuracy
            save_model(args, main_model, model_name=f"Round" + "_best")

        torch.cuda.empty_cache()

    writer.close()