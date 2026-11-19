"""Training model/trainer adapters for non-invasive Stage13 assignment tracing."""
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from chapter4_detection.models.assignment_monitor import AssignmentRecorder,MonitoredAssigner

RECORDER=AssignmentRecorder(enabled=False)

class DiagnosticDetectionModel(DetectionModel):
    def init_criterion(self):
        criterion=v8DetectionLoss(self);criterion.assigner=MonitoredAssigner(criterion.assigner,RECORDER);return criterion

class AssignmentDiagnosticTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=self.set_model_names_for_load(DiagnosticDetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose))
        if weights:model.load(weights)
        return model
