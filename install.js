module.exports = {
  run: [
    // Edit this step to customize the git repository to use
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/VAST-AI-Research/DetailGen3D.git app",
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
    // Edit this step with your custom install commands
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
          "uv pip install accelerate setuptools wheel torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html",
          "uv pip install -r requirements.txt --no-build-isolation",
          "uv pip install gradio pandas==2.0.3" // Had to specify this pandas version for some reason as gradio updated it to a version that is not compatible with the version of numpy in the requirements
        ]
      }
    },
  ]
}