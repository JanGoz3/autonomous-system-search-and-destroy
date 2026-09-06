import torch
from models.driverModel.driver_network import DriverNet

CHECKPOINT_FILE = "unity_sim/driver_training/driver_checkpoint_v4_20_64.pth"
STACKED_VECTORS = 3
STATE_SPACE = 40
ACTION_SPACE = 4

model = DriverNet(in_features=STATE_SPACE * STACKED_VECTORS, out_features=ACTION_SPACE)
checkpoint = torch.load(CHECKPOINT_FILE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
    
model.eval()
dummy_input = torch.randn(1, STATE_SPACE * STACKED_VECTORS)
onnx_filename = "driver_checkpoint_v4_20_64.onnx"

# 2. Export natively via PyTorch
torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["obs_0"],
    # Map exactly to the 4 outputs returned by our new forward() method
    output_names=[
        "continuous_actions",
        "version_number",
        "memory_size",
        "continuous_action_output_shape"
    ],
    dynamic_axes={
        "obs_0": {0: "batch_size"}, 
        "continuous_actions": {0: "batch_size"}
    },
    dynamo=False
)

print(f"Model successfully saved to {onnx_filename}!")
