import io
import numpy as np
import torch
from flask import Flask, request, send_file
from src.split_pipelines import CloudStableDiffusionXLPipeline
from src.quantizers import quantize_wrapper  # <-- Add this import

app = Flask(__name__)

cloud = CloudStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    variant="fp16"
)
cloud.to("cuda")

@app.route('/cloud_prepare', methods=['POST'])
def cloud_prepare():
    # Receive .npz with prompt_embeds, add_text_embeds, add_time_ids, latents
    npz_bytes = io.BytesIO(request.data)
    npz = np.load(npz_bytes)
    prompt_embeds = torch.from_numpy(npz['prompt_embeds']).to("cuda")
    add_text_embeds = torch.from_numpy(npz['add_text_embeds']).to("cuda")
    add_time_ids = torch.from_numpy(npz['add_time_ids']).to("cuda")
    latents = torch.from_numpy(npz['latents']).to("cuda")
    num_inference_steps = int(request.args.get('num_inference_steps', 50))

    cloud.receive_hidden_layer_outputs_and_prepare_for_denoising(
        prompt_embeds=prompt_embeds,
        add_text_embeds=add_text_embeds,
        add_time_ids=add_time_ids,
        latents=latents,
        num_inference_steps=num_inference_steps
    )
    return "OK"

@app.route('/cloud_denoise', methods=['POST'])
def cloud_denoise():
    # Receive step index, cloud_quantize, quantize
    idx = int(request.args['idx'])
    cloud_quantize = request.args.get('cloud_quantize', 'False') == 'True'
    quantize = request.args.get('quantize', 'FP32')

    # Run denoising step
    predicted_noise = cloud.infer_second_pipeline(idx, cloud_quantize, quantize)

    # Quantize predicted_noise before sending
    predicted_noise_npy = predicted_noise.detach().cpu().numpy()
    quantized_noise_npy, quantizer = quantize_wrapper(predicted_noise_npy, quantize)

    # Optionally, you can dequantize back to float32 if you want to keep the edge logic unchanged:
    # predicted_noise = quantized_noise_npy.astype(np.float32)
    # predicted_noise = torch.from_numpy(predicted_noise).to("cpu")

    # Send quantized noise as .npy
    npy_bytes = io.BytesIO()
    np.save(npy_bytes, quantized_noise_npy)
    npy_bytes.seek(0)
    return send_file(npy_bytes, mimetype='application/octet-stream', as_attachment=True, download_name='predicted_noise.npy')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)