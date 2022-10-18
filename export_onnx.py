import torch

import torchcrepe


device = 'cpu'
model = 'full'
wrapper = torchcrepe.CrepeWrapper(model=model)
wrapper.crepe.load_state_dict(torch.load(f'torchcrepe/assets/{model}.pth', map_location=device))
with torch.no_grad():
    frames = torch.rand((2, 1024)).to(device)
    print(frames.shape)
    torch.onnx.export(
        wrapper,
        (
            frames,
        ),
        f'crepe_{model}_wrapped.onnx',
        input_names=['frames'],
        output_names=['probabilities'],
        opset_version=10,
        dynamic_axes={
            'frames': {
                0: 'n_frames'
            }
        }
    )
    print('OK')
