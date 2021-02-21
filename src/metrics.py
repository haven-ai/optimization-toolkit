import torch 
import tqdm
from torch.utils.data import DataLoader
from backpack import extend


@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_name, batch_size=128):
    device = next(model.parameters()).device
    metric_function = get_metric_function(metric_name)
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for batch in tqdm.tqdm(loader):
        images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score
