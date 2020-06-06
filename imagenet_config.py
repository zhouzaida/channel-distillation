class Config:

    log = "./log"  # Path to save log
    checkpoints = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None  # path of evaluated model

    dataset_type = "imagenet"
    train_dataset_path = "./data/ILSVRC2012/train"
    val_dataset_path = "./data/ILSVRC2012/val"

    num_classes = 1000  # Number of classes
    epochs = 100  # Total training epochs
    batch_size = 256
    lr = 0.1  # Learning rate
    num_workers = 10  # Workers of PyTorch to load data

    loss_list = [
        {
            "loss_name": "CELoss",
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "ce_family",
            "loss_rate_decay": "lrdv2",
        },
        {
            "loss_name": "KDLossv2",
            "T": 1,
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "kdv2_family",
            "loss_rate_decay": "lrdv2",
        },
        {
            "loss_name": "CDLoss",
            "loss_rate": 6,
            "factor": 0.9,
            "loss_type": "fd_family",
            "loss_rate_decay": "lrdv2",
        },
    ]
