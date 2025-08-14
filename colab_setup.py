# colab_setup.py
import os
import json
from google.colab import drive

def setup(
    config_dir="/content/drive/MyDrive/config",
    repo_url="https://github.com/cmwinterburn/solar-forecasting.git",
    repo_dir="/content/solar-forecasting",
    use_drive_cache=True,
):
    """
    Prepare a Colab runtime for this project using secrets stored in Drive.
    Expects:
      - oauth_client.json: {"client_id": "...", "client_secret": "..."}
      - default.json: OAuth token file
    """

    # Install DVC with Google Drive support
    os.system('pip -q install "dvc[gdrive]"')

    # Fresh clone
    os.system(f"rm -rf {repo_dir}")
    os.system(f'git clone "{repo_url}" "{repo_dir}"')
    os.chdir(repo_dir)

    # Mount Google Drive
    drive.mount("/content/drive")

    # Load OAuth client credentials
    creds_path = os.path.join(config_dir, "oauth_client.json")
    token_src = os.path.join(config_dir, "default.json")
    assert os.path.exists(creds_path), f"Missing {creds_path}"
    assert os.path.exists(token_src), f"Missing {token_src}"

    with open(creds_path) as f:
        creds = json.load(f)

    CLIENT_ID = creds["client_id"]
    CLIENT_SECRET = creds["client_secret"]

    # Put token where DVC expects it
    CLIENT_DIR = CLIENT_ID
    os.makedirs(f"/root/.cache/pydrive2fs/{CLIENT_DIR}", exist_ok=True)
    os.system(f'cp "{token_src}" "/root/.cache/pydrive2fs/{CLIENT_DIR}/default.json"')

    # Configure DVC with client credentials
    os.system(f'dvc remote modify --local gdrive gdrive_client_id "{CLIENT_ID}"')
    os.system(f'dvc remote modify --local gdrive gdrive_client_secret "{CLIENT_SECRET}"')

    # Optional: persistent DVC cache in Drive
    if use_drive_cache:
        os.system('dvc cache dir /content/drive/MyDrive/.dvc-cache')

    # Remove any stale cache for default GDrive client
    os.system('rm -rf /root/.cache/pydrive2fs/710796635688-iivsgbgsb6uv1fap6635dhvuei09o66c.apps.googleusercontent.com || true')

    # Pull data
    os.system("dvc pull -q")
    os.system("ls -lh data || true")

    print("\n✅ Colab setup complete — repo cloned, Drive mounted, DVC ready.\n")
