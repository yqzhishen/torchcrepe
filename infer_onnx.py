import time

import onnxruntime as ort
import numpy as np
import torch

import torchcrepe


frames = np.load('test.npy')


session = ort.InferenceSession(r'crepe_full.onnx', providers=['DmlExecutionProvider'])
start = time.time()
ort_out = session.run(None, {'frames': frames})
options = ort.RunOptions()
print('ort', time.time() - start)

ort_result = np.array(ort_out[0])
print(np.mean(ort_result))
print(np.var(ort_result))

del session
del ort_out


device = 'cuda'
crepe = torchcrepe.Crepe().to(device)
crepe.load_state_dict(torch.load('torchcrepe/assets/full.pth', map_location=device))
x = torch.FloatTensor(frames)

start = time.time()
with torch.no_grad():
    torch_out = crepe.forward(x.to(device))
print('torch', time.time() - start)

torch_result = torch_out.detach().cpu().numpy()
print(np.mean(torch_result))
print(np.var(torch_result))

print(np.allclose(ort_result, torch_result))
