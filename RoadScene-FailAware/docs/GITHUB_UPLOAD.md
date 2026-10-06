# Upload this repository to GitHub

Only this folder belongs to the GitHub project. Do not upload the parent workspace.
Weights, private inputs and generated outputs are excluded by `.gitignore`.
No remote has been configured and no upload has been performed automatically.

## Option A: GitHub Desktop

1. Create an empty GitHub repository named `RoadScene-FailAware` (do not add a
   generated README/license/gitignore; those already exist here).
2. Add this local folder as a repository in GitHub Desktop.
3. Review the change list: it must not contain weight/data/token files.
4. Commit, then publish/push to the chosen repository.

## Option B: PowerShell / Git

Create an empty repository on GitHub first. Run these commands inside this folder.
Replace `YOUR_GITHUB_USERNAME` with your actual account name:

```powershell
git status --short
py -3.11 scripts/check_repository.py
git add .
git diff --cached --stat
git commit -m "Initial three-model road-scene inference project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/RoadScene-FailAware.git
git push -u origin main
```

Git has already been initialized locally for the prepared folder. No commit,
staging or remote/push was performed for you. If using the ZIP on another
computer, first run `git init` after extraction; `.git/` is not bundled.

Authenticate through GitHub Desktop/Git Credential Manager or your preferred
secure workflow. Never write a PAT/token into a source file or remote URL.

## Option C: Browser upload

Create a repository, choose Add file / Upload files, then upload the **contents**
of this folder. Include dotfiles such as `.gitignore`, `.gitattributes` and
`.github/`. Do not upload the ZIP as the only repository file.

The ZIP intentionally excludes `.git/`, environments, model weights, datasets,
private input images and generated inference output.
