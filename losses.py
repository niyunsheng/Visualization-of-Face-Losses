import torch
import torch.nn as nn

class SinglePolarLoss(nn.Module):
    """single polar loss.
    
    Reference:
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, real_weight=4, num_classes=5, fake_rate=1):
        super(SinglePolarLoss, self).__init__()
        assert num_classes !=2, "[SinglePolarLoss] num_classes shouldn't equal 2"
        ce_weight = torch.Tensor([real_weight,1.0])
        if torch.cuda.is_available():
            ce_weight = ce_weight.cuda()
        self.real_ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=-1)
        self.fake_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.fake_rate = fake_rate
        self.num_classes = num_classes

    def forward(self, output, labels):
        """
        Args:
            output: feature matrix with shape (batch_size, num_classes).
            labels: ground truth labels with shape (batch_size).
        """
        label2 = labels.clone()
        label2[labels>0] = 1
        output_fake = torch.log(torch.sum(torch.exp(output[:,1:]),axis=1)).reshape(-1,1)
        output2 = torch.cat((output[:,0:1],output_fake),dim=1)
        label_fake = labels.clone()
        label_fake -= 1
        if torch.cuda.is_available():
            label2 = label2.cuda()
            label_fake = label_fake.cuda()
            output2 = output2.cuda()
            output = output.cuda()
        real_loss = self.real_ce(output2,label2)
        fake_loss = self.fake_ce(output[:,1:],label_fake)
        loss = real_loss + self.fake_rate * fake_loss
        return loss