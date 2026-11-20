#!/usr/bin/env bash
# Updates the local `public` branch with a sanitized snapshot of the source branch (by default `main`).
#
# `public` is the branch that gets published (for example to GitHub). It has its own, much shorter history: one commit per snapshot, on top of the
# previous one, so every update is a fast-forward of what was already published. The script NEVER pushes; it prints the command at the end.
#
#   tools/update_public_branch.sh [--source main] [--branch public] [--core <path>] [--author "Name <email>"] [--message "text"] [--no-build] [--keep-worktree] [--reset]
#
# The shared code lives in the sibling repository `musicgen-core`. The snapshot vendors its current commit (`git archive HEAD`) under
# `third_party/musicgen-core`, where `Directory.Build.props` finds it as MusicGenCoreDir, so the public branch builds on its own. --core gives the
# path of that repository (default: $MUSICGEN_CORE_DIR, or `../musicgen-core` next to this one); it must be a git repository with no uncommitted
# changes, because the snapshot records its commit. The script stops before touching anything if it is missing.
#
# What a snapshot is:
#   1. `git archive <source>` extracted over a clean checkout of `public` (an orphan branch is created the first time), plus the core under
#      `third_party/musicgen-core` (see above). Every check below runs over both.
#   2. Sanitizing: editor settings (.idea/) are removed and absolute paths of the home directory are replaced (the audio folder by <audio-dir>, any other
#      by `~`).
#   3. LICENSE: kept from the previous snapshot (or copied from $HEARTLIB_DIR/LICENSE the first time), because the source branch does not carry it.
#   4. Checks, any of which aborts without committing: no text or binary file may contain the user name, the home directory, the git identity (name,
#      email and its local part), the host of `origin` or anything listed in $PUBLIC_FORBIDDEN (colon-separated); no private key, GitHub / Hugging Face /
#      AWS / Slack token pattern; and (unless --no-build) the snapshot must build in Release with the CPU backend.
#      --reset starts the branch over as a new single-commit orphan (drops its history): use it only to purge something that was already published,
#      and then publish with a FORCE push. Without it the new snapshot is a normal commit on top of the previous one.
#   5. Commit with the author given by --author (default: the git identity), recording the source commit in the message. Nothing is committed when the
#      snapshot equals the last one.
set -euo pipefail

source_ref=main
branch=public
core=""
author=""
message=""
build=1
keep=0
reset=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_ref=$2; shift 2 ;;
    --branch) branch=$2; shift 2 ;;
    --core) core=$2; shift 2 ;;
    --author) author=$2; shift 2 ;;
    --message) message=$2; shift 2 ;;
    --no-build) build=0; shift ;;
    --keep-worktree) keep=1; shift ;;
    --reset) reset=1; shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

repo=$(git rev-parse --show-toplevel)
cd "$repo"
git rev-parse --verify --quiet "$source_ref^{commit}" >/dev/null || { echo "source '$source_ref' does not exist" >&2; exit 2; }
source_sha=$(git rev-parse --short "$source_ref")

# The shared core: resolved and validated before anything is touched.
core_arg=${core:-${MUSICGEN_CORE_DIR:-$repo/../musicgen-core}}
if ! core=$(cd "$core_arg" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null); then
  echo "the shared core repository was not found or is not a git repository (looked at: $core_arg). Give its path with --core <path> or set MUSICGEN_CORE_DIR" >&2
  exit 2
fi
git -C "$core" rev-parse --verify --quiet "HEAD^{commit}" >/dev/null || { echo "the core repository at $core has no commits" >&2; exit 2; }
if [[ -n $(git -C "$core" status --porcelain) ]]; then
  echo "the core repository at $core has uncommitted changes; commit them first (the snapshot records the core's commit)" >&2
  exit 2
fi
core_sha=$(git -C "$core" rev-parse --short HEAD)

work=${PUBLIC_WORKDIR:-$HOME/.cache/heartmula-public}
rm -rf "$work"; mkdir -p "$(dirname "$work")"
cleanup() { if [[ $keep -eq 0 ]]; then git -C "$repo" worktree remove --force "$work" 2>/dev/null || true; git -C "$repo" worktree prune; fi; }
trap cleanup EXIT

previous_license=""
if [[ $reset -eq 1 ]] && git rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
  echo "--reset: the history of '$branch' ($(git rev-parse --short "$branch")) is replaced; publish with a force push" >&2
  git update-ref -d "refs/heads/$branch"
fi
if git rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
  git worktree add -q "$work" "$branch"
  previous_license=$(git -C "$work" show "HEAD:LICENSE" 2>/dev/null || true)
  git -C "$work" rm -rq --ignore-unmatch . >/dev/null
else
  git worktree add -q --orphan -b "$branch" "$work"
fi

# 1. snapshot
git archive "$source_ref" | tar -x -C "$work"
mkdir -p "$work/third_party/musicgen-core"
git -C "$core" archive HEAD | tar -x -C "$work/third_party/musicgen-core"

# 2. sanitizing
rm -rf "$work/.idea"
find "$work" -name __pycache__ -type d -not -path "$work/.git/*" -exec rm -rf {} + 2>/dev/null || true
find "$work" -name "*.pyc" -type f -not -path "$work/.git/*" -delete
home_escaped=$(printf '%s' "$HOME" | sed 's/[.[\*^$/]/\\&/g')
while IFS= read -r -d '' f; do
  if grep -Iq . "$f" 2>/dev/null && grep -qF "$HOME" "$f"; then
    sed -i -e "s|$HOME/deploy/heartmula-audio|<audio-dir>|g" -e "s|$home_escaped|~|g" "$f"
    echo "sanitized paths in ${f#"$work"/}"
  fi
done < <(find "$work" -path "$work/.git" -prune -o -type f -print0)

# 3. license
if [[ ! -f "$work/LICENSE" ]]; then
  if [[ -n "$previous_license" ]]; then printf '%s\n' "$previous_license" > "$work/LICENSE"
  elif [[ -f "${HEARTLIB_DIR:-$HOME/Proyectos/heartlib}/LICENSE" ]]; then cp "${HEARTLIB_DIR:-$HOME/Proyectos/heartlib}/LICENSE" "$work/LICENSE"
  else echo "no LICENSE in the source, in the previous snapshot or in \$HEARTLIB_DIR" >&2; exit 1; fi
fi

# 4. checks
forbidden=()
add() { [[ ${#1} -ge 4 ]] && forbidden+=("$1"); return 0; }
add "${USER:-$(id -un)}"; add "$HOME"
name=$(git config user.name || true); email=$(git config user.email || true)
add "$name"; add "$email"; add "${email%@*}"
origin_url=$(git remote get-url origin 2>/dev/null || true)
if [[ -n "$origin_url" ]]; then host=${origin_url#*://}; host=${host#*@}; host=${host%%[:/]*}; add "$host"; fi
IFS=':' read -r -a extra <<< "${PUBLIC_FORBIDDEN:-}"; for e in "${extra[@]:-}"; do [[ -n "$e" ]] && add "$e"; done

fail=0
for s in "${forbidden[@]}"; do
  # -a: binary files too (fixtures), so a path embedded in a tensor header would be caught as well
  all=$(grep -raiF --exclude-dir=.git --exclude=.git --exclude-dir=bin --exclude-dir=obj -l -- "$s" "$work" || true)
  if [[ -n "$all" ]]; then echo "FORBIDDEN '$s' found in:" >&2; echo "$all" | sed "s|^$work/|  |" >&2; fail=1; fi
done
secret='-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}'
if grep -raEl --exclude-dir=.git --exclude=.git --exclude-dir=bin --exclude-dir=obj -- "$secret" "$work" >/tmp/public-secret-hits.$$ 2>/dev/null && [[ -s /tmp/public-secret-hits.$$ ]]; then
  echo "SECRET-LIKE content found in:" >&2; sed "s|^$work/|  |" /tmp/public-secret-hits.$$ >&2; fail=1
fi
rm -f /tmp/public-secret-hits.$$
if [[ $fail -ne 0 ]]; then echo "aborting: nothing was committed (the checks above failed)" >&2; exit 1; fi
echo "checks: no personal strings, no secret-like content"

if [[ $build -eq 1 ]]; then
  echo "building the snapshot (Release, CPU backend)..."
  (cd "$work" && dotnet build HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --nologo -v q 2>&1 | tail -3)
fi

# 5. commit
git -C "$work" add -A
if git -C "$work" diff --cached --quiet; then
  echo "no changes: '$branch' already matches $source_ref@$source_sha"
  exit 0
fi
[[ -n "$message" ]] || message="Update the public snapshot to $source_ref@$source_sha with musicgen-core@$core_sha"
commit_args=(-q -m "$message" -m "Source: $source_ref@$source_sha" -m "Core: musicgen-core@$core_sha (vendored in third_party/musicgen-core)")
if [[ -n "$author" ]]; then
  # the committer travels in the commit too: use the same identity, so the git config one (a personal address) does not leak
  [[ "$author" =~ ^(.+)\ \<(.+)\>$ ]] || { echo "--author must look like: Name <email>" >&2; exit 2; }
  commit_args+=(--author "$author")
  export GIT_COMMITTER_NAME="${BASH_REMATCH[1]}" GIT_COMMITTER_EMAIL="${BASH_REMATCH[2]}"
fi
git -C "$work" commit "${commit_args[@]}"
echo "committed on '$branch': $(git -C "$work" log -1 --format='%h %s')"
git -C "$work" diff --stat HEAD~1 HEAD 2>/dev/null | tail -1 || true
echo "push it yourself, for example:  git push <github-remote> $branch:main"
