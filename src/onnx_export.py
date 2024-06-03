import torch
   
from src.model import SegmentationModel
from src.utils import get_root_directory, load_data
   
   
def export_onnx(model_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = SegmentationModel(device)
    model = seg_model.get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 4, 240, 240, 155).to(device)
    torch.onnx.export(model, (dummy_input, ), 'src/onnx/model.onnx', verbose=False)

export_onnx("src/models/best_metric_model.pth")
print("Successfully exported ONNX model!")