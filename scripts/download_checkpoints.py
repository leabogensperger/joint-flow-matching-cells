"""
Download pretrained checkpoints from the Hugging Face Hub into experiments/,
in the exact layout the training scripts already produce (config.yaml +
weights [+ latent_scale.pt for flow models]) so they work directly with
evaluate_vae.py / evaluate_latent_flow_joint.py without any extra setup.

Expects a Hub repo laid out as:
    vae/config.yaml
    vae/best_model.pth
    flow/<drug>/config.yaml
    flow/<drug>/best_model.pt
    flow/<drug>/latent_scale.pt

Usage:
  python scripts/download_checkpoints.py --vae
  python scripts/download_checkpoints.py --drug C7
  python scripts/download_checkpoints.py --all
"""
import argparse
import pathlib
import shutil

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("This script needs huggingface_hub: pip install huggingface_hub")

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DEFAULT_REPO_ID = "leabogensperger/joint-flow-matching-cells"
ALL_DRUGS = ["C7", "CK-666"]


def _fetch(repo_id, remote_path, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=remote_path)
    shutil.copy(cached, local_path)
    print(f"  {remote_path} -> {local_path}")


def download_vae(repo_id):
    out_dir = BASE_DIR / "experiments" / "microscopy_vae" / "pretrained"
    print(f"[VAE] Downloading into {out_dir}")
    for fname in ["config.yaml", "best_model.pth"]:
        _fetch(repo_id, f"vae/{fname}", out_dir / fname)


def download_flow(repo_id, drug):
    out_dir = BASE_DIR / "experiments" / "latent_fm_joint_conc" / f"pretrained_{drug}"
    print(f"[FLOW/{drug}] Downloading into {out_dir}")
    for fname in ["config.yaml", "best_model.pt", "latent_scale.pt"]:
        _fetch(repo_id, f"flow/{drug}/{fname}", out_dir / fname)


def main():
    p = argparse.ArgumentParser(description="Download pretrained checkpoints from the Hugging Face Hub")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face Hub repo id")
    p.add_argument("--vae", action="store_true", help="Download the VAE checkpoint")
    p.add_argument("--drug", type=str, default=None, choices=ALL_DRUGS, help="Download the flow model for one drug")
    p.add_argument("--all", action="store_true", help="Download the VAE and every drug's flow model")
    args = p.parse_args()

    if not (args.vae or args.drug or args.all):
        p.error("Specify --vae, --drug <name>, or --all")

    if args.vae or args.all:
        download_vae(args.repo_id)
    if args.all:
        for drug in ALL_DRUGS:
            download_flow(args.repo_id, drug)
    elif args.drug:
        download_flow(args.repo_id, args.drug)

    print("[DONE]")


if __name__ == "__main__":
    main()
