import argparse, torch
from model_def import UNet

def export(weight_path, onnx_path, input_size=(1, 3, 512, 512)):
    model = UNet()
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and 'state_dict' in state:
        state = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
    model.load_state_dict(state)
    model.eval()
    dummy = torch.zeros(input_size)
    torch.onnx.export(model, dummy, onnx_path, opset_version=12, input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    export(args.weights, args.out)
