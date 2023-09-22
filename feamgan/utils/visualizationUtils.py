import torch
import datetime
import wandb

from feamgan.utils.semanticSegmentationUtils import labelMapToColor

def formatFramesToUnit(data):
    return torch.add(torch.mul(data, 127.5), 127.5).byte()

def formatVideo(vid, data_type="frames", to_cpu=True, dataset_name=None, train_ids=True, vid_index=0):
    vid = vid[vid_index]
    if data_type == "segmentations":
        vid = torch.argmax(vid, keepdim=True, dim=-3)
        vid = labelMapToColor(vid, dataset_name, are_train_ids=train_ids)
    else:
        vid = formatFramesToUnit(vid)
    if to_cpu:
        vid = vid.detach().cpu()
    return vid

def convertTime(t):
    return datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S.%f')

def wandbVisFATEAttention(attention, frame_count, name):
    summary = {}
    attention = torch.mul(attention, 255).byte()
    for i in range(attention.shape[1]):
        attention_map = attention[:,i:i+1]
        shape = attention_map.shape
        print(shape)
        summary[f"attention_map_{shape[-2]}_{shape[-1]}_{name}_{i}"] = wandb.Video(attention_map.detach().cpu(), fps=1, format="gif")
    
    wandb.log(summary, step=frame_count)