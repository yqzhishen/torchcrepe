import torch

import torchcrepe

device = 'cpu'
crepe = torchcrepe.Crepe()
crepe.load_state_dict(torch.load(f'torchcrepe/assets/full.pth', map_location=device))
with torch.no_grad():
    frames = torch.rand((1024, 1024)).to(device)
    print(frames.shape)
    torch.onnx.export(
        crepe,
        (
            frames,
        ),
        'crepe_full.onnx',
        input_names=['frames'],
        opset_version=10,
        dynamic_axes={
            'frames': {
                0: 'n_frames'
            }
        }
    )
    print('OK')
