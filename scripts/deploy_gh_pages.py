"""
Deploys frontend/dist directly to origin/gh-pages using native Git plumbing.
Works cross-platform with zero external dependencies.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "frontend" / "dist"


def run_cmd(cmd, env=None):
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(res.stderr)
        sys.exit(res.returncode)
    return res.stdout.strip()


def deploy():
    if not DIST_DIR.exists() or not (DIST_DIR / "index.html").exists():
        print("Building frontend before deploying...")
        run_cmd(["npm", "run", "build"], env=dict(os.environ, CWD=str(REPO_ROOT / "frontend")))

    # Ensure .nojekyll exists
    nojekyll = DIST_DIR / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.write_text("# Disable Jekyll\n", encoding="utf-8")

    # Isolated index file path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_index_path = tmp_file.name
    if os.path.exists(tmp_index_path):
        os.remove(tmp_index_path)

    try:
        git_env = os.environ.copy()
        git_env["GIT_INDEX_FILE"] = tmp_index_path

        # Stage all files in frontend/dist
        run_cmd(["git", f"--work-tree={DIST_DIR}", "add", "-A"], env=git_env)

        # Write tree
        tree_sha = run_cmd(["git", "write-tree"], env=git_env)
        print(f"Git tree written: {tree_sha}")

        # Commit tree
        commit_sha = run_cmd(
            ["git", "commit-tree", tree_sha, "-m", "Deploy clinical decision support dashboard to GitHub Pages"],
            env=git_env,
        )
        print(f"Git commit created: {commit_sha}")

        # Push to origin gh-pages
        print("Pushing to origin gh-pages...")
        push_out = run_cmd(["git", "push", "origin", f"{commit_sha}:refs/heads/gh-pages", "--force"])
        print(push_out)
        print("\n[SUCCESS] Deployment to gh-pages branch completed successfully!")
    finally:
        if os.path.exists(tmp_index_path):
            try:
                os.remove(tmp_index_path)
            except OSError:
                pass


if __name__ == "__main__":
    deploy()
