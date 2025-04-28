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
    }
  ]
}
