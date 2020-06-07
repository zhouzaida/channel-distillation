def adjust_loss_alpha(alpha, epoch, factor=0.9, loss_type="ce_family", loss_rate_decay="lrdv1", dataset_type="imagenet"):
    """Early Decay Teacher"""

    if dataset_type == "imagenet":
        if loss_rate_decay == "lrdv1":
            return alpha * (factor ** (epoch // 30))
        else:  # lrdv2
            if "ce" in loss_type or "kd" in loss_type:
                return 0 if epoch <= 30 else alpha * (factor ** (epoch // 30))
            else:
                return alpha * (factor ** (epoch // 30))
    else:  # cifar
        if loss_rate_decay == "lrdv1":
            return alpha
        else:  # lrdv2
            if epoch >= 160:
                exponent = 2
            elif epoch >= 60:
                exponent = 1
            else:
                exponent = 0
            if "ce" in loss_type or "kd" in loss_type:
                return 0 if epoch <= 60 else alpha * (factor**exponent)
            else:
                return alpha * (factor**exponent)
