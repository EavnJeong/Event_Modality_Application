# Expanding Event Modality Applications through a Robust CLIP-Based Encoder

![Architecture](figures/fig1.png)

# Data Prepare
## N-ImageNet ([LINK](https://www.image-net.org/), [LINK](https://github.com/82magnolia/n_imagenet))

    Path
    |---ImageNet
    |     |--- train
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |     |--- val
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |---N_ImageNet
    |     |--- extracted_train
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |     |--- extracted_val
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
 
Check the data/prepare.py to put the right path for dataset.

configs/ .txt files should be changed.

 
# Pre training

- Dataset in [N-imagenet, N-imagenet-1000]
- foundation in [ViT-B/32, ViT-L/14]
---
    python main.py --dataset "Dataset" --foundation "foundation"

# FineTuning

- Dataset in [N-imagenet, N-imagenet-1000]
- foundation in [ViT-B/32, ViT-L/14]
- ckpt_path "Pre-training .pt file"
- test_mode : for evaluating only
- ft in [1-shot, 2-shot, 5-shot, all]
---
    python finetune.py --dataset "Dataset" --foundation "foundation" --ckpt_path "ckpt_path" -- ft "ft"

# Download Checkpoints

Checkpoint files are available in HuggingFace [LINK](https://huggingface.co/Eavn/event-clip/tree/main).


    from huggingface_hub import hf_hub_download
    import torch
    import clip
    import torch.nn.functional as F
    import numpy as np
    import cv2
    import torchvision.transforms as transforms


    def generate_event_image(frames, threshold=10):
        frames = np.array(frames)  
        num_frames, height, width, _ = frames.shape
        event_images = []
        
        for i in range(1, num_frames):
            diff = cv2.absdiff(frames[i], frames[i-1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, event_image = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
            event_images.append(event_image)

        return torch.tensor(event_images).sum(dim=0)


    ckpt_path = hf_hub_download(
        repo_id="Eavn/event-clip",      
        filename="vitb.pt", # or vitl.pt
        repo_type="model"               
    )

    model, preprocess = clip.load("ViT-B/32")
    # model, preprocess = clip.load("ViT-L/14")

    state_dict = torch.load(ckpt_path)["checkpoint"]
    new_state_dict = {}
    for key in state_dict.keys():
        if 'encoder_k' in key:
            new_state_dict[key.replace('encoder_k.', '')] = state_dict[key]
    model.load_state_dict(new_state_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    stack_size = 16
    threshold = 10
    clamp = 10
    text = 'Put the Text Here'
    text = clip.tokenize([text]).cuda()

    images = (np.random.rand(32, 224, 224, 3) * 255).astype(np.uint8)
    event = generate_event_image(
        images[:stack_size], 
        threshold=threshold
    )
    if clamp > 0:
        event = torch.clamp(event, min=0, max=clamp)
    event = event / event.max()
    event = torch.stack([event, event, event])
    event = transform(event)
    event = event.cuda().unsqueeze(0)

    logits_per_event, _ = model(event, text)

![Visualization](figures/fig2.png)



## Acknowledgement

This project is based on [Github](https://github.com/Yan98/Event-Camera-Data-Pre-training)
Special thanks to the original authors for their great work.  