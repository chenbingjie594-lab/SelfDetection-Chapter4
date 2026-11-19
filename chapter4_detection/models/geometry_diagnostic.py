from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from chapter4_detection.models.assignment_geometry_monitor import GeometryDetectionLoss
class GeometryDiagnosticModel(DetectionModel):
 def init_criterion(self):return GeometryDetectionLoss(self)
class GeometryDiagnosticTrainer(DetectionTrainer):
 def get_model(self,cfg=None,weights=None,verbose=True):
  model=self.set_model_names_for_load(GeometryDiagnosticModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose))
  if weights:model.load(weights)
  return model
