import os
import sys
import gradio as gr
import numpy as np
import torch
from PIL import Image
import trimesh
import random
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print("DEVICE: ", DEVICE)

DEFAULT_FACE_NUMBER = 100000
MAX_SEED = np.iinfo(np.int32).max
MV_ADAPTER_REPO_URL = "https://github.com/huanngzh/MV-Adapter.git"

RMBG_PRETRAINED_MODEL = "checkpoints/RMBG-1.4"
TRIPOSG_PRETRAINED_MODEL = "checkpoints/TripoSG"

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# The current directory is already the TripoSG repository
TRIPOSG_CODE_DIR = "."

# Clone MV-Adapter if it doesn't exist
MV_ADAPTER_CODE_DIR = "./mv_adapter"
if not os.path.exists(MV_ADAPTER_CODE_DIR):
    os.system(f"git clone {MV_ADAPTER_REPO_URL} {MV_ADAPTER_CODE_DIR}")

# Add paths to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TRIPOSG_CODE_DIR)
sys.path.append(os.path.join(TRIPOSG_CODE_DIR, "scripts"))
sys.path.append(MV_ADAPTER_CODE_DIR)
sys.path.append(os.path.join(MV_ADAPTER_CODE_DIR, "scripts"))


HEADER = """

# 🔮 Image to 3D with [TripoSG](https://github.com/VAST-AI-Research/TripoSG)

## State-of-the-art Open Source 3D Generation Using Large-Scale Rectified Flow Transformers

<p style="font-size: 1.1em;">By <a href="https://www.tripo3d.ai/" style="color: #1E90FF; text-decoration: none; font-weight: bold;">Tripo</a></p>

## 📋 Quick Start Guide:
1. **Upload an image** (single object works best)
2. Click **Generate Shape** to create the 3D mesh
3. Click **Apply Texture** to add textures
4. Use **Download GLB** to save your 3D model
5. Adjust parameters under **Generation Settings** for fine-tuning

Best results come from clean, well-lit images with clear subject isolation. Try it now!

<p style="font-size: 0.9em; margin-top: 10px;">Texture generation powered by <a href="https://github.com/huanngzh/MV-Adapter" style="color: #1E90FF; text-decoration: none;">MV-Adapter</a> - a versatile multi-view adapter for consistent texture generation. Try the <a href="https://huggingface.co/spaces/VAST-AI/MV-Adapter-I2MV-SDXL" style="color: #1E90FF; text-decoration: none;">MV-Adapter demo</a> for multi-view image generation.</p>

"""

# # triposg
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG
# Try to use pre-downloaded models, but download if not available
if not os.path.exists(os.path.join(RMBG_PRETRAINED_MODEL, "model.pth")):
    print("RMBG model not found, downloading now...")
    snapshot_download("briaai/RMBG-1.4", local_dir=RMBG_PRETRAINED_MODEL)
rmbg_net = BriaRMBG.from_pretrained(RMBG_PRETRAINED_MODEL).to(DEVICE)
rmbg_net.eval()
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
# Try to use pre-downloaded models, but download if not available
if not os.path.exists(os.path.join(TRIPOSG_PRETRAINED_MODEL, "diffusion_pytorch_model.safetensors")):
    print("TripoSG model not found, downloading now...")
    snapshot_download("VAST-AI/TripoSG", local_dir=TRIPOSG_PRETRAINED_MODEL)
triposg_pipe = TripoSGPipeline.from_pretrained(TRIPOSG_PRETRAINED_MODEL).to(DEVICE, DTYPE)

# mv adapter
NUM_VIEWS = 6
from mv_adapter.scripts.inference_ig2mv_sdxl import prepare_pipeline, preprocess_image, remove_bg
from mv_adapter.mvadapter.utils import get_orthogonal_camera, tensor_to_image, make_image_grid
from mv_adapter.mvadapter.utils.render import NVDiffRastContextWrapper, load_mesh, render
mv_adapter_pipe = prepare_pipeline(
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_model=None,
    lora_model=None,
    adapter_path="huanngzh/mv-adapter",
    scheduler=None,
    num_views=NUM_VIEWS,
    device=DEVICE,
    dtype=torch.float16,
)
birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
birefnet.to(DEVICE)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, DEVICE)

# Try to use pre-downloaded models, but download if not available
if not os.path.exists("checkpoints/RealESRGAN_x2plus.pth"):
    print("RealESRGAN_x2plus.pth not found, downloading now...")
    try:
        # For Hugging Face downloads, we can use tqdm for progress display
        from tqdm import tqdm

        print("Downloading RealESRGAN_x2plus.pth from Hugging Face...")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="RealESRGAN_x2plus.pth") as t:
            # Create a callback that will update our progress bar
            def progress_callback(progress):
                if progress.total:
                    t.total = progress.total
                t.update(progress.completed - t.n)

            # Download with progress tracking
            hf_hub_download(
                "dtarnow/UPscaler",
                filename="RealESRGAN_x2plus.pth",
                local_dir="checkpoints",
                tqdm_class=lambda *args, **kwargs: progress_callback
            )
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading RealESRGAN_x2plus.pth: {e}")
        print("Trying alternative download method...")
        # Fallback to regular download without progress tracking
        hf_hub_download("dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth", local_dir="checkpoints")
if not os.path.exists("checkpoints/big-lama.pt"):
    print("big-lama.pt not found, downloading now...")
    try:
        import urllib.request
        import time

        # Define a progress bar callback function
        def download_progress_hook(count, block_size, total_size):
            global start_time
            if count == 0:
                global start_time
                start_time = time.time()
                return

            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
            percent = min(int(count * block_size * 100 / total_size), 100)

            # Calculate progress bar width (50 characters)
            bar_width = 50
            filled_width = int(bar_width * percent / 100)
            bar = '█' * filled_width + '-' * (bar_width - filled_width)

            # Print progress bar
            sys.stdout.write(f'\r|{bar}| {percent}% | {progress_size / (1024*1024):.1f} MB | {speed} KB/s')
            sys.stdout.flush()

            # Print a newline when download completes
            if progress_size >= total_size:
                sys.stdout.write('\n')

        # Start the download with progress reporting
        urllib.request.urlretrieve(
            "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            "checkpoints/big-lama.pt",
            reporthook=download_progress_hook
        )
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading big-lama.pt: {e}")
        print("Please run the installation script or download manually.")


def start_session(req: gr.Request):
    save_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(save_dir, exist_ok=True)
    print("start session, mkdir", save_dir)

def end_session(req: gr.Request):
    save_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(save_dir)

def get_random_hex():
    random_bytes = os.urandom(8)
    random_hex = random_bytes.hex()
    return random_hex

def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def run_full(image: str, req: gr.Request):
    seed = 0
    num_inference_steps = 50
    guidance_scale = 7.5
    simplify = True
    target_face_num = DEFAULT_FACE_NUMBER

    image_seg = prepare_image(image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = triposg_pipe(
        image=image_seg,
        generator=torch.Generator(device=triposg_pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).samples[0]
    print("mesh extraction done")
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if simplify:
        print("start simplify")
        from scripts.utils import simplify_mesh
        mesh = simplify_mesh(mesh, target_face_num)

    save_dir = os.path.join(TMP_DIR, "examples")
    os.makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(save_dir, f"triposg_{get_random_hex()}.glb")
    mesh.export(mesh_path)
    print("save to ", mesh_path)

    torch.cuda.empty_cache()

    height, width = 768, 768
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * NUM_VIEWS,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")

    mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)
    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )

    image = Image.open(image)
    image = remove_bg_fn(image)
    image = preprocess_image(image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)

    images = mv_adapter_pipe(
        "high quality",
        height=height,
        width=width,
        num_inference_steps=15,
        guidance_scale=3.0,
        num_images_per_prompt=NUM_VIEWS,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=image,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": 1.0},
        **pipe_kwargs,
    ).images

    torch.cuda.empty_cache()

    mv_image_path = os.path.join(save_dir, f"mv_adapter_{get_random_hex()}.png")
    make_image_grid(images, rows=1).save(mv_image_path)

    from scripts.texture import TexturePipeline, ModProcessConfig
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="checkpoints/big-lama.pt",
        device=DEVICE,
    )

    textured_glb_path = texture_pipe(
        mesh_path=mesh_path,
        save_dir=save_dir,
        save_name=f"texture_mesh_{get_random_hex()}.glb",
        uv_unwarp=True,
        uv_size=4096,
        rgb_path=mv_image_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )

    return image_seg, mesh_path, textured_glb_path


@torch.no_grad()
def run_segmentation(image: str):
    image = prepare_image(image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    return image

@torch.no_grad()
def image_to_3d(
    image: Image.Image,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    simplify: bool,
    target_face_num: int,
    req: gr.Request
):
    outputs = triposg_pipe(
        image=image,
        generator=torch.Generator(device=triposg_pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).samples[0]
    print("mesh extraction done")
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if simplify:
        print("start simplify")
        from scripts.utils import simplify_mesh
        mesh = simplify_mesh(mesh, target_face_num)

    save_dir = os.path.join(TMP_DIR, str(req.session_hash))
    mesh_path = os.path.join(save_dir, f"triposg_{get_random_hex()}.glb")
    mesh.export(mesh_path)
    print("save to ", mesh_path)

    torch.cuda.empty_cache()

    return mesh_path

@torch.no_grad()
def run_texture(image: Image, mesh_path: str, seed: int, req: gr.Request):
    height, width = 768, 768
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * NUM_VIEWS,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")

    mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)
    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )

    image = Image.open(image)
    image = remove_bg_fn(image)
    image = preprocess_image(image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)

    images = mv_adapter_pipe(
        "high quality",
        height=height,
        width=width,
        num_inference_steps=15,
        guidance_scale=3.0,
        num_images_per_prompt=NUM_VIEWS,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=image,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": 1.0},
        **pipe_kwargs,
    ).images

    torch.cuda.empty_cache()

    save_dir = os.path.join(TMP_DIR, str(req.session_hash))
    mv_image_path = os.path.join(save_dir, f"mv_adapter_{get_random_hex()}.png")
    make_image_grid(images, rows=1).save(mv_image_path)

    from scripts.texture import TexturePipeline, ModProcessConfig
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="checkpoints/big-lama.pt",
        device=DEVICE,
    )

    textured_glb_path = texture_pipe(
        mesh_path=mesh_path,
        save_dir=save_dir,
        save_name=f"texture_mesh_{get_random_hex()}.glb",
        uv_unwarp=True,
        uv_size=4096,
        rgb_path=mv_image_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )

    return textured_glb_path


with gr.Blocks(title="TripoSG") as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_prompts = gr.Image(label="Input Image", type="filepath")
                seg_image = gr.Image(
                    label="Segmentation Result", type="pil", format="png", interactive=False
                )

            with gr.Accordion("Generation Settings", open=True):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=0,
                    value=0
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=8,
                    maximum=50,
                    step=1,
                    value=50,
                )
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=7.0,
                )

                with gr.Row():
                    reduce_face = gr.Checkbox(label="Simplify Mesh", value=True)
                    target_face_num = gr.Slider(maximum=1000000, minimum=10000, value=DEFAULT_FACE_NUMBER, label="Target Face Number")

                gen_button = gr.Button("Generate Shape", variant="primary")
                gen_texture_button = gr.Button("Apply Texture", interactive=False)

        with gr.Column():
            model_output = gr.Model3D(label="Generated GLB", interactive=False)
            textured_model_output = gr.Model3D(label="Textured GLB", interactive=False)

    with gr.Row():
        # Use a simpler approach for examples - just load the images without preprocessing
        examples = gr.Examples(
            examples=[
                f"assets/example_data/{image}"
                for image in os.listdir(f"assets/example_data")
                if image.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ],
            inputs=[image_prompts],
            # Don't specify outputs or function to avoid automatic processing
            # The user will need to click "Generate Shape" after selecting an example
            cache_examples=False,
        )

    gen_button.click(
        run_segmentation,
        inputs=[image_prompts],
        outputs=[seg_image]
    ).then(
        get_random_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            seg_image,
            seed,
            num_inference_steps,
            guidance_scale,
            reduce_face,
            target_face_num
        ],
        outputs=[model_output]
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_texture_button])

    gen_texture_button.click(
        run_texture,
        inputs=[image_prompts, model_output, seed],
        outputs=[textured_model_output]
    )

    demo.load(start_session)
    demo.unload(end_session)

demo.launch(
    share=False,
    max_threads=8,         # Limit number of threads to prevent excessive resource usage
    show_error=True        # Show detailed error messages for debugging
)