import copy,torch
from ultralytics.utils.tal import TaskAlignedAssigner
from chapter4_detection.models.assignment_monitor import AssignmentRecorder,MonitoredAssigner

def sample():
 torch.manual_seed(7);scores=torch.rand(1,84,2);boxes=torch.rand(1,84,4)*64;boxes[...,2:]=boxes[...,:2]+boxes[...,2:].abs()+2
 yy,xx=torch.meshgrid(torch.arange(8)+.5,torch.arange(8)+.5,indexing='ij');a=torch.cat((torch.stack((xx,yy),-1).reshape(-1,2)*8,torch.stack(torch.meshgrid(torch.arange(4)+.5,torch.arange(4)+.5,indexing='ij'),-1).flip(-1).reshape(-1,2)*16,torch.stack(torch.meshgrid(torch.arange(2)+.5,torch.arange(2)+.5,indexing='ij'),-1).flip(-1).reshape(-1,2)*32))
 labels=torch.tensor([[[0.],[1.]]]);gt=torch.tensor([[[20.,20.,29.,28.],[35.,35.,55.,58.]]]);mask=torch.ones(1,2,1,dtype=torch.bool);return scores,boxes,a,labels,gt,mask
def test_monitor_outputs_loss_and_gradient_identical():
 x=sample();base=TaskAlignedAssigner(topk=10,num_classes=2,alpha=.5,beta=6.,stride=[8,16,32]);ref=base(*x);rec=AssignmentRecorder();got=MonitoredAssigner(copy.deepcopy(base),rec)(*x)
 for a,b in zip(ref,got):assert torch.equal(a,b)
 p1=torch.randn(1,84,2,requires_grad=True);p2=p1.detach().clone().requires_grad_(True)
 l1=torch.nn.functional.binary_cross_entropy_with_logits(p1,ref[2]);l2=torch.nn.functional.binary_cross_entropy_with_logits(p2,got[2]);l1.backward();l2.backward()
 assert torch.equal(l1,l2);assert torch.equal(p1.grad,p2.grad);assert len(rec.records)==2
