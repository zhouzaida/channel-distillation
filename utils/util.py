def adjust_loss_alpha(alpha, epoch, factor=0.9, loss_type="ce_family", loss_rate_decay="lrdv1"):
    """动态调整蒸馏的比例

    loss_type: 损失函数的类型
        "ce_family": loss输入为student的pred以及label
        "kd_family": loss输入为student的pred、teacher的pred
        "kdv2_family": loss输入为student的pred、teacher的pred以及label
        "fd_family": loss输入为student的feature、teacher的feature
    loss_rate_decay: 衰减策略
        "lrdv1": 一开始就有ce或者kd
        "lrdv2": 前30epoch没有ce或者kd
    """
    if loss_rate_decay not in ["lrdv1", "lrdv2", "lrdv3", "lrdv4", "lrdv5"]:
        raise Exception("loss_rate_decay error")

    if loss_type not in ["ce_family", "kd_family", "kdv2_family", "fd_family"]:
        raise Exception("loss type error")

    if loss_rate_decay == "lrdv1":
        return alpha * (factor ** (epoch // 30))
    elif loss_rate_decay == "lrdv2":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 30 else alpha * (factor ** (epoch // 30))
        else:
            return alpha * (factor ** (epoch // 30))
    elif loss_rate_decay == "lrdv5":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 60 else alpha
        else:
            if epoch >= 160:
                return alpha * (factor**3)
            elif epoch >= 120:
                return alpha * (factor**2)
            elif epoch >= 60:
                return alpha * (factor**1)
            else:
                return alpha
