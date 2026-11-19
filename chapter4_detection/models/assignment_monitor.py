"""Non-invasive TaskAlignedAssigner monitor for Stage13."""
from __future__ import annotations
from dataclasses import dataclass,field
import torch
from torch import nn

@dataclass
class AssignmentRecorder:
    records:list[dict]=field(default_factory=list)
    epoch:int|None=None
    enabled:bool=True

class MonitoredAssigner(nn.Module):
    """Pass through the installed assigner exactly, then record detached diagnostics."""
    def __init__(self,base:nn.Module,recorder:AssignmentRecorder):super().__init__();self.base=base;self.recorder=recorder
    @property
    def topk(self):return self.base.topk
    def forward(self,pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes,mask_gt):
        out=self.base(pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes,mask_gt)
        if self.recorder.enabled:self._record(pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes,mask_gt,out)
        return out
    @torch.no_grad()
    def _record(self,pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes,mask_gt,out):
        # Recompute diagnostics on detached tensors only; returned tensors above are untouched.
        inside=self.base.select_candidates_in_gts(anc_points,gt_bboxes,mask_gt)
        align,overlap=self.base.get_box_metrics(pd_scores,pd_bboxes,gt_labels,gt_bboxes,inside*mask_gt)
        labels,boxes,scores,fg,idx=out
        strides=torch.tensor(self.base.stride,device=anc_points.device);counts=[];start=0
        # Infer level boundaries from unique anchor spacing is ambiguous; assign by stride-specific regular grids.
        # For square training input the count is (imgsz/s)^2, derived from maximum anchor extent.
        side=float(anc_points[:,0].max()+min(self.base.stride)/2)
        for s in self.base.stride:counts.append(int(round(side/s))**2)
        for b in range(gt_bboxes.shape[0]):
          valid=int(mask_gt[b].sum())
          for g in range(valid):
            pos=fg[b]&(idx[b]==g);ious=overlap[b,g,pos];rec={'epoch':self.recorder.epoch,'batch_index':b,'gt_index':g,'class_id':int(gt_labels[b,g,0]),'gt_width':float(gt_bboxes[b,g,2]-gt_bboxes[b,g,0]),'gt_height':float(gt_bboxes[b,g,3]-gt_bboxes[b,g,1]),'inside_count':int(inside[b,g].sum()),'positive_count':int(pos.sum()),'positive_iou_mean':float(ious.mean()) if len(ious) else None,'positive_iou_max':float(ious.max()) if len(ious) else None,'target_score_sum':float(scores[b,pos].sum())}
            st=0
            for li,n in enumerate(counts):rec[f'positive_p{li+3}']=int(pos[st:st+n].sum());st+=n
            self.recorder.records.append(rec)
