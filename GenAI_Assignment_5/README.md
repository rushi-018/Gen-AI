# GenAI Lab 5 — Generating Creative Artwork Using Stable Diffusion

## Assignment Overview

This is **Assignment 5** for the Generative AI Lab course. The objective is to explore and demonstrate how **Stable Diffusion**, a state-of-the-art text-to-image generative AI model, can be used to create creative artwork from textual descriptions (prompts).

### Presented By

- **Rushiraj Suwarnkar**
- **Aditya Deshmukh**

---

## Project Structure

```
GenAI_Assignment_5/
├── Creative_Artwork_using_Stable_Diffusion (1).ipynb   # Main notebook
├── Presentation - Generating Creative Artwork.pdf       # Presentation slides
├── stable_diffusion_art.png                             # Default generated artwork
├── user input image1.png                                # User-prompted image 1
├── user input image2.png                                # User-prompted image 2
└── README.md                                            # This file
```

---

## What Does the Notebook Do?

1. **Installs Dependencies** — Installs `diffusers`, `transformers`, `accelerate`, `safetensors`, and `torch`.
2. **Loads the Model** — Downloads and loads the `runwayml/stable-diffusion-v1-5` pre-trained pipeline with FP16 precision on a CUDA GPU.
3. **Generates a Default Image** — Produces an image from a built-in prompt:  
   _"a surreal painting of a floating city in the clouds, dreamy, cinematic lighting, ultra detailed"_
4. **Interactive User Prompts** — Allows users to enter their own text prompts and generates images in real time.
5. **Saves Outputs** — Each generated image is saved as a PNG file for review.

---

## How to Run

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (required for `torch.float16` on GPU)
- ~5 GB disk space for model weights

### Steps

1. Open the notebook in VS Code or Jupyter.
2. Run the first cell to install required packages:
   ```bash
   pip install -q diffusers transformers accelerate safetensors torch
   ```
3. Run the second cell to load the Stable Diffusion pipeline and generate the default image.
4. Run subsequent cells to enter your own prompts and generate custom artwork.

---

## Key Parameters

| Parameter        | Value                            | Description                                                                |
| ---------------- | -------------------------------- | -------------------------------------------------------------------------- |
| `model_id`       | `runwayml/stable-diffusion-v1-5` | Pre-trained Stable Diffusion checkpoint                                    |
| `torch_dtype`    | `torch.float16`                  | Half-precision for faster inference and lower VRAM usage                   |
| `guidance_scale` | `7.5`                            | Controls how closely the image follows the prompt (higher = more faithful) |

---

## Extra Knowledge: Stable Diffusion In-Depth

### What is Stable Diffusion?

Stable Diffusion is an open-source **latent diffusion model** (LDM) developed by Stability AI, CompVis (LMU Munich), and Runway. It generates high-quality images from text descriptions by operating in a compressed latent space rather than directly on pixel data, making it significantly more efficient than earlier diffusion models.

### How It Works — The Core Pipeline

Stable Diffusion consists of **three main components**:

1. **Text Encoder (CLIP)**  
   The text prompt is tokenized and passed through a frozen CLIP (Contrastive Language–Image Pre-training) text encoder. This converts the prompt into a high-dimensional embedding that captures its semantic meaning.

2. **U-Net (Denoising Network)**  
   A U-Net neural network iteratively denoises a random latent tensor, guided by the text embedding. Over many timesteps (typically 20–50), it progressively transforms pure noise into a meaningful latent representation of the image. This is the core "diffusion" process.

3. **VAE Decoder (Variational Autoencoder)**  
   The final denoised latent tensor is decoded back into pixel space by the VAE decoder, producing the output image (usually 512×512 pixels for v1.5).

```
Text Prompt ──► CLIP Text Encoder ──► Text Embedding
                                            │
Random Noise ──► U-Net (iterative denoising, guided by text embedding) ──► Denoised Latent
                                                                                │
                                                            VAE Decoder ──► Final Image
```

### The Diffusion Process

- **Forward diffusion**: Gaussian noise is gradually added to an image over many steps until it becomes pure noise. This is used during training.
- **Reverse diffusion**: Starting from random noise, the model learns to remove noise step-by-step, recovering a clean image that matches the conditioning (text prompt). This is used during inference.

### Why "Latent" Diffusion?

Unlike pixel-space diffusion models (e.g., DALL·E 2 uses a pixel-level decoder), Stable Diffusion operates in a **compressed latent space** (64×64×4 instead of 512×512×3). This reduces computation by ~48× while preserving image quality, which is why it can run on consumer GPUs.

### Key Concepts

| Concept                  | Explanation                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Guidance Scale (CFG)** | Classifier-Free Guidance — a higher value (e.g., 7–12) makes the output more closely match the prompt but may reduce diversity. A value of 1.0 means no guidance. |
| **Negative Prompts**     | Text describing what you _don't_ want in the image (e.g., "blurry, low quality"). Helps steer the model away from undesired outputs.                              |
| **Scheduler / Sampler**  | The algorithm used for the denoising steps (e.g., DDPM, DDIM, Euler, DPM-Solver). Different schedulers offer trade-offs between speed and quality.                |
| **Inference Steps**      | Number of denoising iterations (default ~50). More steps generally improve quality but increase generation time. 20–30 steps are often sufficient.                |
| **Seed**                 | The random seed for the initial noise tensor. Using the same seed + prompt produces the same image, enabling reproducibility.                                     |
| **img2img**              | A variant where an existing image (instead of pure noise) is used as the starting point, allowing style transfer or image editing.                                |
| **Inpainting**           | Selectively regenerating parts of an image using a mask, useful for editing specific regions.                                                                     |

### Stable Diffusion Versions

| Version         | Release  | Key Improvements                                                               |
| --------------- | -------- | ------------------------------------------------------------------------------ |
| **v1.4**        | Aug 2022 | First public release, 512×512                                                  |
| **v1.5**        | Oct 2022 | Improved fine-tuning, used in this assignment                                  |
| **v2.0 / v2.1** | Nov 2022 | New text encoder (OpenCLIP), 768×768 support, depth-guided generation          |
| **SDXL**        | Jul 2023 | 1024×1024 native resolution, two-stage architecture (base + refiner)           |
| **SD 3.0**      | 2024     | Multimodal Diffusion Transformer (MMDiT) architecture, improved text rendering |

### Tips for Better Prompts

- **Be specific**: "a photorealistic portrait of an astronaut riding a horse on Mars, golden hour lighting, 8K" works better than "astronaut on horse".
- **Use style keywords**: Include art styles like _"oil painting"_, _"watercolor"_, _"digital art"_, _"anime style"_, _"cinematic lighting"_.
- **Use quality boosters**: Add _"highly detailed"_, _"4K"_, _"masterpiece"_, _"trending on ArtStation"_.
- **Leverage negative prompts**: Exclude _"blurry, out of focus, deformed, ugly, low quality"_ to improve results.

### Practical Considerations

- **VRAM**: Stable Diffusion v1.5 requires ~4 GB VRAM with FP16. Use `torch.float16` (as in this notebook) to halve memory usage.
- **CPU Fallback**: It can run on CPU but is extremely slow (minutes per image vs. seconds on GPU).
- **Safety Checker**: The pipeline includes an optional NSFW safety checker that can be disabled if needed.
- **Commercial Use**: Stable Diffusion is released under the CreativeML Open RAIL-M license, which permits commercial use with some restrictions.

---

## Libraries Used

| Library        | Purpose                                            |
| -------------- | -------------------------------------------------- |
| `diffusers`    | Hugging Face library for diffusion model pipelines |
| `transformers` | Tokenizer and text encoder (CLIP)                  |
| `accelerate`   | Efficient model loading and device placement       |
| `safetensors`  | Safe and fast model weight serialization           |
| `torch`        | PyTorch deep learning framework                    |

---

## References

- [Stable Diffusion v1.5 — Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Paper)](https://arxiv.org/abs/2112.10752)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [CLIP: Learning Transferable Visual Models (Paper)](https://arxiv.org/abs/2103.00020)
