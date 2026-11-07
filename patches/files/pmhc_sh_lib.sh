#!/usr/bin/env bash
# Shared helpers of the mHC family's .sh scenario checks (p011/p019/p022/p025_sh_check.sh). Source it; never run it.
# Needs bash, python3 (no torch), sha256sum or shasum. Scratch VLLM_ROOT trees only: no real vLLM install is touched.
# Caller sets before sourcing: PDIR (patches dir), TREE (the v0.32.0 `vllm` dir), WORK.
set -u
pmhc_init() {
  rm -rf "$WORK"; mkdir -p "$WORK/roots"
  P="$WORK/patches"; cp -R "$PDIR" "$P"; rm -rf "$P/files/__pycache__"
  REALPY="$(command -v python3)"; pass=0; fail=0
}
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# pin <script-glob-prefix> <NAME>: the pinned sha NAME= in the scratch copy of NNN-*.sh
pin() { sed -n "s/^$2=\([0-9a-f]\{64\}\)$/\1/p" "$P"/$1-*.sh | head -1; }
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
# mkroot <name> <rel>...: a scratch VLLM_ROOT seeded with the v0.32.0 originals of the listed relative paths
mkroot() {
  local n=$1; shift; local r="$WORK/roots/$n"; rm -rf "$r" "$r.log"; mkdir -p "$r"
  local f; for f in "$@"; do mkdir -p "$r/$(dirname "$f")"; cp "$TREE/$f" "$r/$f"; done
  echo "$r"
}
# run <script> <root> [VAR=val ...] -> exit code on stdout; stdout+stderr of the script appended to <root>.log
run() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
# a python3 shim that fails (or corrupts the target of) one applier invocation: shim_fail <dir> <applier-name> [corrupt]
shim_make() {
  local d=$1 pat=$2 mode=${3:-fail}; mkdir -p "$d"
  pat="${pat// /\\ }"      # escape spaces for the case pattern
  if [ "$mode" = fail ]; then
    cat > "$d/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *$pat*) echo "shim: simulated applier failure" >&2; exit 1 ;; esac
exec "$REALPY" "\$@"
EOF
  else
    cat > "$d/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *$pat*) printf '\n# unexpected\n' >> "\${@: -1}"; exit 0 ;; esac
exec "$REALPY" "\$@"
EOF
  fi
  chmod +x "$d/python3"
}
pmhc_summary() { echo "---- $pass passed, $fail failed"; [ "$fail" = 0 ] && echo "ALL PASS"; [ "$fail" = 0 ]; }
