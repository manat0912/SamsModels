module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        env: {
          GIT_LFS_SKIP_SMUDGE: "1"
        },
        message: [
          "git lfs install",
          "git clone https://github.com/manat0912/Sam3Video.git app",
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          xformers: true,
          triton: true,
          sageattention: true
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python merge_models.py",
        venv: "env",
        path: "app"
      }
    }
  ]
}
