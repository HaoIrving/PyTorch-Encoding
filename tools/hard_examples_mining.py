'''
first method:TOPK loss
思路：将每批次训练样本中的loss的最大的K%个样本记录下来，然后对这些困难样本做数据增强
这个方法用于于目标检测
链接：https://blog.csdn.net/u010165147/article/details/97105166
'''


'''
second method:focal loss
对于大的loss的样本给与较大的权重，较小的loss给与较小的权重
说明链接：https://blog.csdn.net/CaiDaoqing/article/details/90457197?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-4-90457197.nonecase&utm_term=分割loss&spm=1000.2123.3001.4430
'''
import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


    def focal_loss(self, output, target, alpha, gamma, OHEM_percent):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)
 
        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
 
        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss
 
        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()
