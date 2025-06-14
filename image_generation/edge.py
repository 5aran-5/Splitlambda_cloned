import io
import numpy as np
import torch
import requests
from PIL import Image
from src.split_pipelines import EdgeStableDiffusionXLPipeline

CLOUD_IP = "192.168.x.x"  # <-- Replace with your GPU server's IP

def main():
    # User input
    prompt = input("Enter your prompt: ")
    num_inference_steps = 50

    # Load edge pipeline
    edge = EdgeStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    edge.to("cpu")

    # First subpipeline
    prompt_embeds, add_text_embeds, add_time_ids, latents = edge.infer_first_pipeline(
        prompt=prompt, num_inference_steps=num_inference_steps
    )

    # Send to cloud for preparation
    npz_bytes = io.BytesIO()
    np.savez_compressed(
        npz_bytes,
        prompt_embeds=prompt_embeds.cpu().numpy(),
        add_text_embeds=add_text_embeds.cpu().numpy(),
        add_time_ids=add_time_ids.cpu().numpy(),
        latents=latents.cpu().numpy()
    )
    npz_bytes.seek(0)
    resp = requests.post(
        f"http://{CLOUD_IP}:5000/cloud_prepare?num_inference_steps={num_inference_steps}",
        data=npz_bytes.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    assert resp.text == "OK"

    # Denoising loop
    for idx in range(num_inference_steps):
        print(f"Step {idx+1}/{num_inference_steps}")
        # Request predicted noise from cloud
        resp = requests.post(
            f"http://{CLOUD_IP}:5000/cloud_denoise?idx={idx}&cloud_quantize=False&quantize=FP32"
        )
        npy_bytes = io.BytesIO(resp.content)
        predicted_noise = np.load(npy_bytes)
        predicted_noise = torch.from_numpy(predicted_noise).to("cpu")

        # Third subpipeline on edge
        edge.infer_third_pipeline(idx, predicted_noise)

        # Optionally decode and show/save image every N steps
        if (idx + 1) % 10 == 0 or (idx + 1) == num_inference_steps:
            image = edge.decode_image()
            image.save(f"result_step_{idx+1}.png")
            print(f"Saved result_step_{idx+1}.png")

if __name__ == '__main__':
    main()