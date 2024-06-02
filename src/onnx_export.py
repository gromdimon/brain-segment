import torch
   
from src.model import SegmentationModel
from src.utils import get_root_directory, load_data
   
#import wandb
   
   
def export_onnx(model_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = SegmentationModel(device)
    model = seg_model.get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 4, 240, 240, 160)
    torch.onnx.export(model, (dummy_input, ), 'src/onnx/model.onnx')

export_onnx("src/models/last_model.pth")
