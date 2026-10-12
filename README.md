ComfyUI-HEART



Native HEART architecture support for ComfyUI using the original HEART architecture source and checkpoint files.



Status: Experimental but functional.



Overview



This project enables native execution of the HEART (Hybrid Efficient Attention with Rank-factorized Bias Transformer) architecture within the ComfyUI Python environment.



The original HEART architecture source was adapted and isolated from its training ecosystem dependencies, allowing successful model construction, checkpoint loading, and inference without relying on ONNX.



Verified Functionality



The following components have been successfully validated:



✅ HEART architecture imports correctly

✅ HEART model instantiates successfully

✅ heart\_4x\_otf\_v2.safetensors loads successfully

✅ State dictionary loads without errors

✅ Forward inference executes correctly

✅ Verified 4× upscaling (64×64 → 256×256)

Validation Output

Plain Text

HEART imported

CHECKPOINT LOADED

torch.Size(\[1, 3, 256, 256])

Background



Standard ComfyUI installations cannot load:



Plain Text

heart\_4x\_otf\_v2.safetensors



through the built-in UpscaleModelLoader because the HEART architecture is not registered in the bundled Spandrel architecture registry.



This project uses the original HEART architecture implementation and adapts the required dependencies so the architecture can be instantiated and executed directly within ComfyUI's Python environment.



Credits

HEART



Author



Philip Hofmann



Project



HEART — Hybrid Efficient Attention with Rank-factorized Bias Transformer

Additional Projects

traiNNer-redux

Spandrel

Project Structure

Plain Text

ComfyUI-HEART/

├── heart\_arch.py

├── hat\_iln\_arch.py

└── arch\_util.py

Verified Model

Plain Text

heart\_4x\_otf\_v2.safetensors

Model Metadata

JSON

{

"embed\_dim": 180,

"depths": \[6, 6, 6, 6, 6, 6],

"num\_heads": \[6, 6, 6, 6, 6, 6],

"window\_size": 32,

"upscale": 4,

"rank": 8,

"rib\_hidden\_dim": 32,

"rib\_n\_freqs": 10

}

Validation Commands

Import Test

PowerShell

\& "C:\\Users\\jglsp\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe" -c "import heart\_arch; print('HEART imported')"



Expected output:



Plain Text

HEART imported

Architecture Construction Test

PowerShell

\& "C:\\Users\\jglsp\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe" -c "from heart\_arch import heart; m=heart(); print(type(m).\_\_name\_\_)"



Expected output:



Plain Text

HEART

Checkpoint Loading Test

PowerShell

\& "C:\\Users\\jglsp\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe" -c "from heart\_arch import heart; from safetensors.torch import load\_file; m=heart(); sd=load\_file(r'C:\\Users\\jglsp\\ComfyUI-Shared\\models\\upscale\_models\\heart\_4x\_otf\_v2.safetensors'); m.load\_state\_dict(sd); print('CHECKPOINT LOADED')"



Expected output:



Plain Text

CHECKPOINT LOADED

 

Forward Inference Test

PowerShell

\& "C:\\Users\\jglsp\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe" -c "from heart\_arch import heart; from safetensors.torch import load\_file; import torch; m=heart(); m.load\_state\_dict(load\_file(r'C:\\Users\\jglsp\\ComfyUI-Shared\\models\\upscale\_models\\heart\_4x\_otf\_v2.safetensors')); x=torch.randn(1,3,64,64); y=m(x); print(y.shape)"

 



Expected output:



Plain Text

torch.Size(\[1, 3, 256, 256])

Current Status



The following functionality has been confirmed:



Architecture import

Model instantiation

Checkpoint compatibility

Weight loading

Successful inference

Correct 4× output generation

Remaining Work



To fully integrate HEART into ComfyUI, the following is still required:



Create a dedicated ComfyUI custom node

Implement model loading from the ComfyUI interface

Implement image-to-image upscaling workflow nodes

Add UI integration and node registration

License



This project does not claim ownership of the HEART architecture, source code, or model weights.



Please follow the original licenses and usage terms provided by the HEART project and all upstream dependencies.



This repository provides compatibility and integration modifications only.



Acknowledgements



Special thanks to:



Philip Hofmann for the HEART architecture

the-database for traiNNer-redux

chaiNNer-org for Spandrel

The ComfyUI community for continued experimentation and architecture support development



Proof of concept status: Successfully executes native HEART inference without ONNX. 🚀

