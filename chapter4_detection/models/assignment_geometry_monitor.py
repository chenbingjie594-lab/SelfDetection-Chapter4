"""Non-invasive per-positive TAL/DFL geometry recorder using real feature shapes."""
from __future__ import annotations
from dataclasses import dataclass,field
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors

@dataclass
class GeometryRecorder:
    records:list[dict]=field(default_factory=list);epoch:int|None=None;enabled:bool=False;assignment_call:int=0

RECORDER=GeometryRecorder()

def representable_iou_max(anchor,gt,max_distance):
    """Exact axis-aligned optimum: independently choose each reachable edge nearest the GT edge."""
    ax,ay=anchor;x1,y1,x2,y2=gt;mx,my=max_distance
    px1=min(max(x1,ax-mx),ax);px2=max(min(x2,ax+mx),ax);py1=min(max(y1,ay-my),ay);py2=max(min(y2,ay+my),ay)
    ix=max(0,min(px2,x2)-max(px1,x1));iy=max(0,min(py2,y2)-max(py1,y1));inter=ix*iy;pa=max(0,px2-px1)*max(0,py2-py1);ga=(x2-x1)*(y2-y1);return inter/(pa+ga-inter+1e-12)

class GeometryDetectionLoss(v8DetectionLoss):
    """Return the untouched parent loss, then recompute detached diagnostics."""
    def get_assigned_targets_and_loss(self,preds,batch):
        result=super().get_assigned_targets_and_loss(preds,batch)
        if RECORDER.enabled:self._record(preds,batch)
        return result
    @torch.no_grad()
    def _record(self,preds,batch):
        RECORDER.assignment_call += 1
        assignment_call = RECORDER.assignment_call
        pd,pcls=preds['boxes'].permute(0,2,1).contiguous(),preds['scores'].permute(0,2,1).contiguous();anchors,strides=make_anchors(preds['feats'],self.stride,.5);bs=pcls.shape[0];imgsz=torch.tensor(preds['feats'][0].shape[2:],device=self.device,dtype=pcls.dtype)*self.stride[0]
        t=torch.cat((batch['batch_idx'].view(-1,1),batch['cls'].view(-1,1),batch['bboxes']),1);t=self.preprocess(t.to(self.device),bs,imgsz[[1,0,1,0]]);gl,gb=t.split((1,4),2);mask=gb.sum(2,keepdim=True).gt_(0);pb=self.bbox_decode(anchors,pd);_,tb,ts,fg,gi=self.assigner(pcls.sigmoid(),(pb*strides).type(gb.dtype),anchors*strides,gl,gb,mask)
        counts=[int(f.shape[2]*f.shape[3]) for f in preds['feats']];ends=torch.tensor(counts,device=self.device).cumsum(0)
        for b,j in fg.nonzero(as_tuple=False):
            g=int(gi[b,j]);box=gb[b,g];a=anchors[j]*strides[j];s=float(strides[j]);signed=torch.stack((a[0]-box[0],a[1]-box[1],box[2]-a[0],box[3]-a[1]))/s;clipped=signed.clamp(0,self.reg_max-1-.01);logp=F.log_softmax(pd[b,j].view(4,self.reg_max),1);tl=clipped.long();tr=tl+1;edge_loss=-(logp[range(4),tl]*(tr-clipped)+logp[range(4),tr]*(clipped-tl));w=float(ts[b,j].sum());inside=bool((signed>0).all());level=int((j>=ends).sum())+3
            ciou=float(bbox_iou(pb[b,j:j+1],(box/s).view(1,4),xywh=False,CIoU=True).squeeze());pred=pb[b,j]*s;gx=(box[0]+box[2])/2;gy=(box[1]+box[3])/2;px=(pred[0]+pred[2])/2;py=(pred[1]+pred[3])/2;gw=box[2]-box[0];gh=box[3]-box[1]
            ew=torch.maximum(gw,torch.tensor(16.,device=box.device));eh=torch.maximum(gh,torch.tensor(16.,device=box.device));eff=torch.stack((gx-ew/2,gy-eh/2,gx+ew/2,gy+eh/2))
            r={'epoch':RECORDER.epoch,'assignment_call':assignment_call,'batch_index':int(b),'gt_index':g,'class_id':int(gl[b,g,0]),'anchor_index':int(j),'level':f'P{level}','stride':s,'anchor_x':float(a[0]),'anchor_y':float(a[1]),'gt_x1':float(box[0]),'gt_y1':float(box[1]),'gt_x2':float(box[2]),'gt_y2':float(box[3]),'effective_x1':float(eff[0]),'effective_y1':float(eff[1]),'effective_x2':float(eff[2]),'effective_y2':float(eff[3]),'gt_width':float(gw),'gt_height':float(gh),'is_tiny':bool(min(gw,gh)<8),'inside_original':inside,'extension_only':not inside,'pred_ciou':ciou,'pred_iou_clamped':float(self.assigner.iou_calculation(box.view(1,4),pred.view(1,4))[0]),'target_score':w,'dfl_loss_unweighted':float(edge_loss.mean()),'dfl_loss_weighted':float(edge_loss.mean()*w),'ciou_loss_weighted':float((1-ciou)*w),'center_error_px':float(torch.hypot(px-gx,py-gy)),'width_error_px':float(pred[2]-pred[0]-gw),'height_error_px':float(pred[3]-pred[1]-gh),'representable_iou_max':float(representable_iou_max((float(a[0]),float(a[1])),tuple(float(x) for x in box),(s*(self.reg_max-1-.01),s*(self.reg_max-1-.01))))}
            for k,n in enumerate(('l','t','r','b')):r[f'signed_{n}']=float(signed[k]);r[f'clipped_{n}']=float(clipped[k]);r[f'dfl_edge_loss_{n}']=float(edge_loss[k]);r[f'clipped_{n}_flag']=bool(signed[k]<0 or signed[k]>self.reg_max-1-.01)
            RECORDER.records.append(r)
