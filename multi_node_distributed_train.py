import argparse
import os
import time
from functools import partial

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler

# import wandb
from data import RSDataset
from peft import LoraConfig, get_peft_model


def setup(
    local_rank,
    local_size=8,
    node_rank=0,
    world_size=8,
    master_addr="localhost",
    master_port="12355",
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    world_rank = local_rank + node_rank * local_size
    if world_rank > 0:
        time.sleep(10)
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return world_rank


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=800,
    )
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    processor,
):
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=0,
            sampler=val_sampler,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders


def evaluate_model(
    world_rank, model, val_loaders, device, processor, epoch, max_val_step, log_file_dir
):
    # Evaluation phase
    model.eval()
    for val_name, val_loader in val_loaders.items():
        val_loss = 0
        with torch.no_grad():
            val_step_count = 0
            for batch in tqdm(
                val_loader,
                desc=f"Evaluation on {val_name} at Epoch {epoch+1}",
                disable=world_rank != 0,
            ):
                val_step_count += 1
                inputs, answers = batch

                # Prepare the input and target tensors
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                    truncation=True,
                    max_length=800,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()
                if val_step_count > max_val_step:
                    break

        if world_rank == 0:
            avg_val_loss = val_loss / val_step_count
            print_message = f"Epoch {epoch+1} - Average Validation Loss ({val_name}): {avg_val_loss}"
            print(print_message)
            with open(log_file_dir + "log_file.txt", "a", encoding="utf8") as f:
                f.write(print_message + "\n")

    model.train()


def train_model(
    local_rank,
    local_size,
    node_rank,
    world_size,
    master_addr,
    master_port,
    checkpoint_path,
    dataset_name,
    json_file=None,
    batch_size=6,
    num_workers=6,
    use_lora=False,
    epochs=10,
    lr=1e-6,
    save_steps=10,
    run_name=None,
    max_val_step=1000,
):
    world_rank = setup(
        local_rank, local_size, node_rank, world_size, master_addr, master_port
    )
    device = torch.device(f"cuda:{local_rank}")
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Load the dataset based on the dataset_name argument
    if dataset_name == "Falcon_SFT":
        train_dataset = RSDataset(split="train", json_file=json_file)
        val_datasets = {
            "Falcon_SFT": RSDataset(split="test", json_file=json_file),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    if use_lora:
        TARGET_MODULES = [
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "linear",
            "Conv2d",
            "lm_head",
            "fc2",
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create DataLoaders
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        num_workers,
        processor,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(train_loader)
    num_training_steps = epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0
    start_time = time.time()
    log_file_dir = f"./model_checkpoints/{run_name}/"
    if world_rank == 0:
        os.makedirs(log_file_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        local_step = 0
        for batch in tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}/{epochs}",
            disable=world_rank != 0,
        ):
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1
            local_step += 1

            if global_step % 5 == 0:
                if world_rank == 0:
                    end_time = time.time()
                    time_per_step = (end_time - start_time) / 5
                    eta = ((num_training_steps - global_step) * time_per_step) / 3600

                    now = time.localtime()
                    print_message = time.strftime(
                        "%Y-%m-%d %H:%M:%S", now
                    ) + ", epoch/total_epoch: {}/{}, process: {:.2f}%, loss: {:.5f}, ETA: {:.2f}h".format(
                        epoch + 1, epochs, local_step / steps_per_epoch * 100, loss, eta
                    )
                    print("")
                    print(print_message)
                    with open(log_file_dir + "log_file.txt", "a", encoding="utf8") as f:
                        f.write(print_message + "\n")
                    start_time = time.time()

            if global_step % save_steps == 0:
                if world_rank == 0:  # Only the main process saves the checkpoint
                    output_dir = f"./model_checkpoints/{run_name}/step_{global_step}"
                    os.makedirs(output_dir, exist_ok=True)
                    model.module.save_pretrained(output_dir)
                    processor.save_pretrained(output_dir)

        if world_rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            print_message = (
                "-----------epoch: {}, avg_train_loss: {}-------------".format(
                    epoch + 1, avg_train_loss
                )
            )
            print("")
            print(print_message)
            with open(log_file_dir + "log_file.txt", "a", encoding="utf8") as f:
                f.write(print_message + "\n")

            output_dir = f"./model_checkpoints/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

        evaluate_model(
            world_rank,
            model,
            val_loaders,
            device,
            processor,
            epoch,
            max_val_step,
            log_file_dir,
        )

    cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Train Falcon model on specified dataset"
    )
    parser.add_argument(
        "--local_size", type=int, default=8, help="number of gpus per node"
    )
    parser.add_argument("--node_rank", type=int, default=0, help="rank of current node")
    parser.add_argument(
        "--world_size", type=int, default=8, help="total number of used gpus"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="ip of master node"
    )
    parser.add_argument(
        "--master_port", type=str, default="12355", help="port of master node"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="Falcon-Single-Instruction-Large",
        help="path to pretrained checkpoint",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to train on"
    )
    parser.add_argument(
        "--label_json", type=str, required=True, help="Path to label json file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for training (4large or 8base)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="number of workers for training",
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA if this flag is passed"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Number of steps between evaluations",
    )
    parser.add_argument("--run_name", type=str, default="ethan", help="Run name")
    parser.add_argument(
        "--max_val_step",
        type=int,
        default=1000,
        help="Maximum number of step to evaluate on during validation",
    )
    args = parser.parse_args()

    mp.spawn(
        train_model,
        args=(
            args.local_size,
            args.node_rank,
            args.world_size,
            args.master_addr,
            args.master_port,
            args.checkpoint_path,
            args.dataset,
            args.label_json,
            args.batch_size,
            args.num_workers,
            args.use_lora,
            args.epochs,
            args.lr,
            args.save_steps,
            args.run_name,
            args.max_val_step,
        ),
        nprocs=args.local_size,
        join=True,
    )


if __name__ == "__main__":
    main()
