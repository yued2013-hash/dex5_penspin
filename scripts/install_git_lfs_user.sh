#!/usr/bin/env bash
set -euo pipefail

VERSION="3.7.1"
ARCHIVE="git-lfs-linux-amd64-v${VERSION}.tar.gz"
EXPECTED_SHA256="1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08"
INSTALL_DIR="${HOME}/.local/bin"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local
  git lfs version
  exit 0
fi
[[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]] || {
  echo "This installer currently supports Linux x86_64 only." >&2
  exit 1
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
curl -fL --retry 5 \
  "https://github.com/git-lfs/git-lfs/releases/download/v${VERSION}/${ARCHIVE}" \
  -o "${tmp_dir}/${ARCHIVE}"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${tmp_dir}/${ARCHIVE}" | sha256sum --check
tar -xzf "${tmp_dir}/${ARCHIVE}" -C "${tmp_dir}"
mkdir -p "${INSTALL_DIR}"
install -m 0755 "${tmp_dir}/git-lfs-${VERSION}/git-lfs" "${INSTALL_DIR}/git-lfs"
export PATH="${INSTALL_DIR}:${PATH}"
git lfs install --local
git lfs version
