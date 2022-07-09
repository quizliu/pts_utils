import coremltools as ct
import torch

model_input_path = 'yolov5s6.pt'
img_size = 640
model_saved_path = 'yolov5s6.mlmodel'

model = torch.load(model_input_path, map_location=torch.device('cpu'))['model'].float()
model.eval()
input = torch.rand(1, 3, img_size, img_size)
model = torch.jit.trace(model, input)
model = ct.convert(
    model,
    # convert_to="mlprogram",
    inputs = [ct.TensorType(name="input", shape=input.shape)]
)
model.save(model_saved_path)