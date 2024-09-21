
import torch

from torch.autograd import Function
from sklearn.metrics import confusion_matrix

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    target[target > 0] = 1
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
        
    return s / (i + 1)

def dice_coeff_fold(input, target):
    """Dice coeff for batches"""
    target[target > 0] = 1
    dsc_fold = []
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        dsc = DiceCoeff().forward(c[0], c[1])
        s = s + dsc
        dsc_fold.append(dsc)
    dsc_fold.append(s / (i + 1))
    return dsc_fold

def get_dice_loss(input, target):
   
    return 1- dice_coeff(input, target)  

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SRS, GTS, eps = 1e-6):
    GTS[GTS > 0] = 1
    corr = torch.sum(SRS==GTS)
    tensor_size = SRS.size(0)*SRS.size(1)*SRS.size(2)*SRS.size(3)
    acc = float(corr)/float(tensor_size + eps)

    return acc
def get_precision(SRS, GTS, eps = 1e-6):
    """Precision for batches"""
    if SRS.is_cuda:
        PC = torch.FloatTensor(1).cuda().zero_()
    else:
        PC = torch.FloatTensor(1).zero_()
    
    
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    GTS[GTS > 0] = 1
    
    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        # TP : True Positive
        # FP : False Positive
        TP = torch.sum( (SR == 1) & (GT == 1) )
        FP = torch.sum( (SR == 1) & (GT == 0) )

        PC += (float(TP)) / (float(TP + FP) + eps)

    return PC / (i + 1)

def get_sensitivity(SRS,GTS, eps = 1e-6):
    # Sensitivity == Recall
    if SRS.is_cuda:
        SE = torch.FloatTensor(1).cuda().zero_()
    else:
        SE = torch.FloatTensor(1).zero_()

    # GT = GT == torch.max(GT)
    GTS[GTS > 0] = 1
    
    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        # TP : True Positive
        # FN : False Negative
        TP = torch.sum( (SR == 1) & (GT == 1) )
        FN = torch.sum( (SR == 0) & (GT == 1) )
        
        SE = SE + float(TP) / ( float(TP + FN) + eps )

    return SE / (i + 1)

def get_specificity(SRS, GTS, eps = 1e-6):
    """Specificity for batches"""
    if SRS.is_cuda:
        SP = torch.FloatTensor(1).cuda().zero_()
    else:
        SP = torch.FloatTensor(1).zero_()
        
    # GT = GT == torch.max(GT)
    GTS[GTS > 0] = 1

    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        # TN : True Negative
        # FP : False Positive
        TN = torch.sum( (SR == 0) & (GT == 0) )
        FP = torch.sum( (SR == 1) & (GT == 0) )

        SP = SP + float(TN) / ( float(TN + FP) + eps )       
    
    return SP / (i + 1)


def get_F1(SRS,GTS, eps = 1e-6):

    SE = get_sensitivity(SRS, GTS)
    PC = get_precision(SRS, GTS)

    F1 = 2 * SE * PC / (SE + PC + eps)
    
    return F1

def get_F1_2(SRS,GTS, eps = 1e-6):
   
    if SRS.is_cuda:
        F1 = torch.FloatTensor(1).cuda().zero_()
    else:
        F1 = torch.FloatTensor(1).zero_()
    
    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        
        TP = torch.sum( (SR == 1) & (GT == 1) )
        FP = torch.sum( (SR == 1) & (GT == 0) )
        FN = torch.sum( (SR == 0) & (GT == 1) )
        
        SE = float(TP) / ( float(TP + FN) + eps )
        PC = get_precision(SRS, GTS)

        F1 = F1 +  2 * SE * PC / (SE + PC + eps)
    
    return F1 / (i + 1)

def get_DSC(SRS, GTS, eps = 1e-6): 
    """Dice coeff for batches"""

    DSC = 0
    GTS[GTS > 0] = 1
    # GTS = np.where(GTS>0, 1, 0)
    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        
        # Inter = torch.dot(SR.view(-1),  GT.view(-1))
        # union = torch.sum(SR) + torch.sum(GT) + eps
        Inter = torch.sum((SR == 1) & (GT == 1))
        union = torch.sum(SR) + torch.sum(GT) + eps
        
        DSC = DSC + (2*float(Inter) )/ float(union)

    return DSC / (i + 1)

def get_JS(SRS, GTS, eps = 1e-6): 
    # JS : Jaccard similarity
    if SRS.is_cuda:
        JS = torch.FloatTensor(1).cuda().zero_()
    else:
        JS = torch.FloatTensor(1).zero_()
        
    GTS[GTS > 0] = 1
    
    for i, (SR, GT) in enumerate(zip(SRS, GTS)):
        
        Inter = torch.sum((SR == 1) & (GT == 1))
        union = torch.sum((SR == 1) | (GT == 1)) + eps
        
        JS = JS + float(Inter) / float(union)
    
    return JS / (i + 1)

def get_TP(SR, GT):
    
    GT[GT > 0] = 1
    return torch.sum((SR == 1) & (GT == 1))

def get_FP(SR, GT):
    
    GT[GT > 0] = 1
    return torch.sum((SR == 1) & (GT == 0))

def get_TN(SR, GT):
    
    GT[GT > 0] = 1
    return torch.sum((SR == 0) & (GT == 0))

def get_FN(SR, GT):
    
    GT[GT > 0] = 1
    return torch.sum((SR == 0) & (GT == 1))

def get_Confusion_Matrix(SR, GT):
        
    TP = get_TP(SR, GT)
    FP = get_FP(SR, GT)
    TN = get_TN(SR, GT)
    FN = get_FN(SR, GT)

    return TP, FP, TN, FN
  