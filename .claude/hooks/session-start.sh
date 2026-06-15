#!/bin/bash
set -euo pipefail

# SessionStart hook for TangoFlux Internals on Claude Code on the web.
#
# Does two things:
#   1. Installs the local package + dev tools (modal, ruff).
#   2. Re-applies the certifi CA patch that lets Modal's gRPC client connect
#      through this environment's TLS-intercepting egress proxy.
#
# Heavy ML deps (torch, transformers, TangoFlux, librosa, ...) are NOT installed
# locally on purpose -- they live inside the remote Modal image. Local Python
# only dispatches jobs.

# Only run inside the remote (web) environment. Locally this is a no-op.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "[session-start] Installing local package (modal + ruff)..."
python3 -m pip install -q -e ".[dev]"

# --- Modal connectivity fix ---------------------------------------------------
# Modal's grpclib client verifies TLS against certifi's bundle, which does NOT
# include this environment's egress proxy CA. Without this patch every Modal
# call fails with: ssl.SSLCertVerificationError ... self-signed certificate in
# certificate chain. We append the locally-trusted custom CAs to certifi.
CUSTOM_CA_DIR="/usr/local/share/ca-certificates"
MARKER="# claude-code: appended egress proxy CAs"
CERTIFI_BUNDLE="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"

if [ -n "$CERTIFI_BUNDLE" ] && [ -d "$CUSTOM_CA_DIR" ]; then
  if grep -qF "$MARKER" "$CERTIFI_BUNDLE" 2>/dev/null; then
    echo "[session-start] certifi bundle already patched; skipping."
  else
    shopt -s nullglob
    cas=("$CUSTOM_CA_DIR"/*.crt)
    if [ ${#cas[@]} -gt 0 ]; then
      # Append each cert separately with surrounding newlines. The source .crt
      # files have no trailing newline, so a bare `cat file1 file2` would glue
      # one cert's END line to the next cert's BEGIN line and produce an invalid
      # PEM bundle (X509 PEM lib error). Per-file echoes guarantee separation.
      printf '\n%s\n' "$MARKER" >> "$CERTIFI_BUNDLE"
      for c in "${cas[@]}"; do
        printf '\n# %s\n' "$(basename "$c")" >> "$CERTIFI_BUNDLE"
        cat "$c" >> "$CERTIFI_BUNDLE"
        printf '\n' >> "$CERTIFI_BUNDLE"
      done
      echo "[session-start] Appended ${#cas[@]} egress CA cert(s) to certifi bundle."
    else
      echo "[session-start] No custom CA certs found; skipping certifi patch."
    fi
  fi
fi

# --- Modal auth note ----------------------------------------------------------
# Tokens are NOT committed. Modal reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from
# the environment automatically, so set them as environment secrets in your
# Claude Code web environment for `modal` commands to authenticate.
if [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ]; then
  echo "[session-start] MODAL_TOKEN_ID/SECRET detected in env; Modal will use them."
else
  echo "[session-start] No MODAL_TOKEN_ID/SECRET in env. Set them as environment"
  echo "                secrets (or run 'modal token set ... --profile X --no-verify') to authenticate."
fi

echo "[session-start] Done."
