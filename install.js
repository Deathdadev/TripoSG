module.exports = {
  run: [
    // Clone the TripoSG repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/VAST-AI-Research/TripoSG.git app",
        ]
      }
    },
    // Clone the MV-Adapter repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/huanngzh/MV-Adapter.git app/mv_adapter",
        ]
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    // Copy app.py to the app directory
    {
      method: "fs.copy",
      params: {
        src: "app.py",
        dest: "app/app.py"
      }
    },
    // Download required model files
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["briaai/RMBG-1.4"],
        "local-dir": "checkpoints/RMBG-1.4"
      }
    },
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["VAST-AI/TripoSG"],
        "local-dir": "checkpoints/TripoSG"
      }
    },
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["dtarnow/UPscaler", "RealESRGAN_x2plus.pth"],
        "local-dir": "checkpoints"
      }
    },
    {
      method: "fs.download",
      params: {
        uri: "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        dir: "app/checkpoints"
      }
    },
    // Install dependencies
    {
      method: "shell.run",
      params: {
	    build: true,
        env: {
          USE_NINJA: 0,
          DISTUTILS_USE_SDK: 1,
        },
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "uv pip install accelerate setuptools wheel",
          "uv pip install -r requirements.txt --no-build-isolation",
          "uv pip install gradio pandas==2.0.3 matplotlib", // Had to specify this pandas version for some reason as gradio updated it to a version that is not compatible with the version of numpy in the requirements
          "uv pip install spandrel==0.4.1 --no-deps",
          "uv pip install -r mv_adapter/requirements.txt"
        ]
      }
    }
  ]
}