#!/usr/bin/env bash
# deploy.sh — compile check + push to GitHub (triggers Streamlit Cloud deploy)
set -e

PYTHON="C:/Users/ryanr/AppData/Local/Programs/Python/Python312/python.exe"
REPO="/c/3PC-Financials"

echo "==> Compiling engine.py..."
"$PYTHON" -m py_compile "$REPO/engine.py" && echo "    ✓ engine.py"

echo "==> Compiling app.py..."
"$PYTHON" -m py_compile "$REPO/app.py" && echo "    ✓ app.py"

echo "==> Pushing to GitHub..."
cd "$REPO"
git push origin main

echo ""
echo "✓ Deploy complete — Streamlit Cloud will update in ~30 seconds"
