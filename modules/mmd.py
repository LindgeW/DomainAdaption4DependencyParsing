import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def coral(x):
    n = x.size(0)
    id_row = x.data.new_ones(1, n)
    sum_column = torch.mm(id_row, x)
    mean_column = torch.div(sum_column, n)
    mean_mean = torch.mm(mean_column.t(), mean_column)
    d_d = torch.mm(x.t(), x)
    coral_result = (d_d - mean_mean) / (n-1)
    return coral_result


def coral2(source, target):
    d = source.data.size(1)
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t().contiguous() @ xm
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t().contiguous() @ xmt
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


def corarl_loss(source, target):
    d = source.data.size(1)
    # source covariance
    #xm = torch.mean(source, 0, keepdim=True) - source
    #xc = xm.t().contiguous() @ xm
    xc = coral(source)
    # target covariance
    #xmt = torch.mean(target, 0, keepdim=True) - target
    #xct = xmt.t().contiguous() @ xmt
    xct = coral(target)
    # frobenius norm between source and target
    #loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss






