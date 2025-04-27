import os
import random
import tempfile
from typing import Any, List, Union

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
import trimesh
from skimage import measure

from detailgen3d.pipelines.pipeline_detailgen3d import DetailGen3DPipeline
from detailgen3d.inference_utils import generate_dense_grid_points

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MARKDOWN = """
## Generating geometry details guided by reference image with [DetailGen3D](https://detailgen3d.github.io/DetailGen3D/)
1. Upload a detailed image of the frontal view and a coarse model. Then clik "Generate Details" to generate the refined result.
2. If you find the generated 3D scene satisfactory, download it by clicking the "Download GLB" button.
3. If you want the refine result to be more consistent with the image, please manually increase the CFG strength.
"""
EXAMPLES = [
    [
        "assets/image/503d193a-1b9b-4685-b05f-00ac82f93d7b.png",
        "assets/model/503d193a-1b9b-4685-b05f-00ac82f93d7b.glb",
        42,
        False,
    ],
    [
        "assets/image/34933195-9c2c-4271-8d31-a28bc5348b7a.png",
        "assets/model/34933195-9c2c-4271-8d31-a28bc5348b7a.glb",
        2131379184,
        False,
    ],
    [
        "assets/image/a5d09c66-1617-465c-aec9-431f48d9a7e1.png",
        "assets/model/a5d09c66-1617-465c-aec9-431f48d9a7e1.glb",
        42,
        False,
    ],
    [
        "assets/image/cb7e6c4a-b4dd-483c-9789-3d4887ee7434.png",
        "assets/model/cb7e6c4a-b4dd-483c-9789-3d4887ee7434.glb",
        42,
        False,
    ],
    [
        "assets/image/e799e6b4-3b47-40e0-befb-b156af8758ad.png",
        "assets/model/e799e6b4-3b47-40e0-befb-b156af8758ad.glb",
        42,
        False,
    ],
    [
        "assets/image/100.png",
        "assets/model/100.glb",
        42,
        False,
    ],
]


os.makedirs(TMP_DIR, exist_ok=True)

local_dir = "pretrained_weights/DetailGen3D"
snapshot_download(repo_id="VAST-AI/DetailGen3D", local_dir=local_dir)
pipeline = DetailGen3DPipeline.from_pretrained(
    local_dir
).to(DEVICE, dtype=DTYPE)


def load_mesh(mesh_path, num_pc=20480):
    mesh = trimesh.load(mesh_path,force="mesh")

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(1.9 / scale)

    surface, face_indices = trimesh.sample.sample_surface(mesh, 1000000,)
    normal = mesh.face_normals[face_indices]

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()

    return surface

@torch.no_grad()
@torch.autocast(device_type=DEVICE)
def run_detailgen3d(
    pipeline,
    image,
    mesh,
    seed,
    num_inference_steps,
    guidance_scale,
):
    surface = load_mesh(mesh)
    # image = Image.open(image).convert("RGB")

    batch_size = 1

    # sample query points for decoding
    box_min = np.array([-1.005, -1.005, -1.005])
    box_max = np.array([1.005, 1.005, 1.005])
    sampled_points, grid_size, bbox_size = generate_dense_grid_points(
        bbox_min=box_min, bbox_max=box_max, octree_depth=8, indexing="ij"
    )
    sampled_points = torch.FloatTensor(sampled_points).to(DEVICE, dtype=DTYPE)
    sampled_points = sampled_points.unsqueeze(0).repeat(batch_size, 1, 1)

    # inference pipeline
    sample = pipeline.vae.encode(surface).latent_dist.sample()
    occ = pipeline(image, latents=sample, sampled_points=sampled_points, guidance_scale=guidance_scale, noise_aug_level=0, num_inference_steps=num_inference_steps).samples[0]

    # marching cubes
    grid_logits = occ.view(grid_size).cpu().numpy()
    vertices, faces, normals, _ = measure.marching_cubes(
        grid_logits, 0, method="lewiner"
    )
    vertices = vertices / grid_size * bbox_size + box_min
    mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
    return mesh

@torch.no_grad()
@torch.autocast(device_type=DEVICE)
def run_refinement(
    rgb_image: Any,
    mesh: Any,
    seed: int,
    randomize_seed: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    scene = run_detailgen3d(
        pipeline,
        rgb_image,
        mesh,
        seed,
        num_inference_steps,
        guidance_scale,
    )

    _, tmp_path = tempfile.mkstemp(suffix=".glb", prefix="detailgen3d_", dir=TMP_DIR)
    scene.export(tmp_path)

    torch.cuda.empty_cache()

    return tmp_path, tmp_path, seed

# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_prompts = gr.Image(label="Example Image", type="pil")

            with gr.Accordion("Generation Settings", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=50.0,
                    step=0.1,
                    value=10.0,
                )
            gen_button = gr.Button("Generate Details", variant="primary")

        with gr.Column():
            mesh = gr.Model3D(label="Input Coarse Model",camera_position=(90,90,3))

            model_output = gr.Model3D(label="Generated GLB", camera_position=(90,90,3))
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)

    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            fn=run_refinement,
            inputs=[image_prompts, mesh, seed, randomize_seed],
            outputs=[model_output, download_glb, seed],
            cache_examples=False,
        )

    gen_button.click(
        run_refinement,
        inputs=[
            image_prompts,
            mesh,
            seed,
            randomize_seed,
            num_inference_steps,
            guidance_scale,
        ],
        outputs=[model_output, download_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_glb])


demo.launch()