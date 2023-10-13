import torch
import torch.nn.functional as F

import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score

@torch.no_grad()
def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.clip.hidden_state_text_projection(model.clip.text_model(texts).last_hidden_state)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
    print(zeroshot_weights.shape)
    return zeroshot_weights

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                logits = pred_logits(
                    model, images, classifier, trials=1
                )            
            true.append(target.cpu())
            pred.append(logits.float().cpu())
    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}

@torch.no_grad()
def pred_logits(model, images, text_embs, trials=1, bs=32):
    vae, clip, unet, noise_scheduler = model.vae, model.clip, model.unet, model.noise_scheduler
    device = images.device
    clip_mean = clip.module.mean if hasattr(clip, "module") else clip.mean
    clip_std = clip.module.std if hasattr(clip, "module") else clip.std
    x = (images+1)/2
    x = (x - clip_mean) / clip_std 
    image_out = clip.vision_model(x)

    latents = vae.encode(images).latent_dist.sample()
    latents = latents * 0.18215
    nbims = images.shape[0]
    nbclasses = text_embs.shape[0]
    nbprompts = text_embs.shape[1]
    nbtexts = nbclasses * nbprompts
    latents = latents.repeat(nbtexts, 1, 1, 1)#nbtexts*nbims,...
    text_embs = text_embs.repeat(nbims, 1, 1, 1)#nbims*nbclasses,nbprompts, length, hidden_dim
    
    nbims_nbclasses, nbprompts, length, hidden_dim = text_embs.shape

    text_embs = text_embs.view(nbims, nbtexts, length, hidden_dim)
    text_embs = text_embs.transpose(0, 1)
    text_embs = text_embs.reshape(nbtexts * nbims, length, hidden_dim)
    losses = []
    for t in range(trials):
        noise = torch.randn_like(latents).to(device)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (nbims * nbtexts,),
            device=device,
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        
        nps = []
        for i in range(0, len(noisy_latents), bs):
            (noise_pred,) = unet(
                noisy_latents[i:i+bs], timesteps[i:i+bs], text_embs[i:i+bs], return_dict=False
            )
            nps.append(noise_pred)       
        noise_pred = torch.cat(nps, dim=0)
        text_to_image_loss = F.mse_loss(noise_pred, target, reduction="none").mean(dim=(1, 2,3))
        text_to_image_loss = text_to_image_loss.view(nbtexts, nbims).T
        losses.append(text_to_image_loss.cpu())
    L = torch.stack(losses)
    L = L.view(trials, nbims, nbclasses, nbprompts)
    L = L.mean(dim=(0, 3))
    return -L