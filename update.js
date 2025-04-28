module.exports = {
  run: [
    // Update the Pinokio repository
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Update the TripoSG repository
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    // Update the MV-Adapter repository
    {
      method: "shell.run",
      params: {
        path: "app/mv_adapter",
        message: "git pull"
      }
    },
    // Copy the updated app.py to the app directory
    {
      method: "fs.copy",
      params: {
        src: "app.py",
        dest: "app/app.py"
      }
    },
    // Make sure directories exist
    {
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "mkdir -p checkpoints",
          "mkdir -p tmp"
        ]
      }
    },
    // Update required model files
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
    }
  ]
}
