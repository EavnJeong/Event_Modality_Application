import model.clip as clip


def get_foundation_model(args):
    if args.foundation == 'ViT-B/32':
        model, preprocess = clip.load("ViT-B/32")
    elif args.foundation == 'ViT-L/14':
        model, preprocess = clip.load("ViT-L/14")
    else:
        raise ValueError(f'Foundation model {args.foundation} not supported')

    model = model.float()
    
    return model, preprocess