import datetime
import logging
import time

import torch
import torch.distributed as dist

from lib.utils.comm import get_world_size

from .inference import inference


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    meters,
    device,
    log_period,
    checkpoint_period,
    evaluate_period,
    arguments,
):
    logger = logging.getLogger("CompFashion.trainer")
    logger.info("Start training")

    max_epoch = arguments["max_epoch"]
    epoch = arguments["epoch"]
    max_iter = max_epoch * len(data_loader)
    iteration = arguments["iteration"]
    distributed = arguments["distributed"]

    best_topk = 0.0
    start_training_time = time.time()
    end = time.time()

    while epoch < max_epoch:
        if distributed:
            data_loader.sampler.set_epoch(epoch)

        epoch += 1
        model.train()
        arguments["epoch"] = epoch

        for step, batch_data in enumerate(data_loader):
            data_time = time.time() - end
            inner_iter = step
            iteration += 1
            arguments["iteration"] = iteration

            imgs_query = batch_data["source_images"].to(device)
            mod_texts = batch_data["text"].to(device)
            text_lengths = batch_data["text_lengths"].to(device)
            imgs_target = batch_data["target_images"].to(device)

            loss_dict = model.compute_loss(
                imgs_query, mod_texts, text_lengths, imgs_target
            )
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if inner_iter % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{inner_iter}/{num_iter}]",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        inner_iter=inner_iter,
                        num_iter=len(data_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                    )
                )

        #             break

        scheduler.step()

        if epoch % evaluate_period == 0:
            topk = 0.0
            for loader in data_loader_val:
                topk += inference(model, loader, save_data=False)
            topk /= 3.0
            meters.update(topk=topk)
            if topk > best_topk:
                best_topk = topk
                checkpointer.save("best", **arguments)

        if epoch % checkpoint_period == 0:
            checkpointer.save("epoch_{:d}".format(epoch), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
