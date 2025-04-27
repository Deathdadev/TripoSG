module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    method: "fs.copy",
    params: {
      from: "app.py",
      to: "app/app.py",
      overwrite: true
    }
  }]
}
