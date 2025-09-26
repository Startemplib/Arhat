####### pdf_rs(Portal_document_Format Resampleing) #######

##### Dependency

# qpdf_path = r"C:\2SilverHand\qpdf\bin\qpdf.exe"
# gs_path = r"C:\2SilverHand\ghostpdl\gs10.05.1\bin\gswin64c.exe"

### pdf_paths          (list[str | Path])     PDFs to process; order is preserved in the return.

### wts                (int)                  Save mode:
#                                              - 0 : Save alongside source (auto `_N` suffix). If `output_dir`
#                                                   is given, save there instead (still uses `_N` to avoid collisions).
#                                              - 1 : Replace original atomically (write temp, then move over on success).

### output_dir         (str | Path | None)    Target directory when `wts == 0`. None -> next to source file(s).

### qpdf_opts          (dict | None)          QPDF options override. Keys = numeric shorthands or full names.
#                                               Values:
#                                               • None   -> include as a flag (no value)
#                                               • str/int-> include as `key=value`
#                                               • False  -> drop/disable this option

### gs_opts            (dict | None)          Ghostscript options override. Same rules as `qpdf_opts`.
#                                             Common: {"-dPDFSETTINGS": "/screen|/ebook|/printer|/prepress"}.

### keep_temp          (bool)                 Keep intermediate working directory (QPDF outputs, temp files).
#                                             If True, the function will print/log the temp dir path.

### qpdf_path          (str | Path)           Path to `qpdf` executable.
### gs_path            (str | Path)           Path to Ghostscript CLI (e.g., `gswin64c.exe` on Windows).
### Returns            (list[Path])           Output PDF paths, same order as `pdf_paths`.


import subprocess
from pathlib import Path
import re
import shutil
import tempfile


def pdf_rs(
    pdf_paths,
    wts=0,
    output_dir=None,
    qpdf_opts=None,  # dict: numeric keys (preferred) or full option names
    gs_opts=None,  # dict: numeric keys (preferred) or full option names
    keep_temp=False,
    qpdf_path=r"C:\2SilverHand\qpdf\bin\qpdf.exe",
    gs_path=r"C:\2SilverHand\ghostpdl\gs10.05.1\bin\gswin64c.exe",
):

    # ---------- Shorthand maps (code -> real option name) ----------
    qpdf_map = {
        1: "--warning-exit-0",
        2: "--stream-data",
        3: "--object-streams",
        4: "--optimize-images",
        5: "--jpeg-quality",
    }
    gs_map = {
        1: "-sDEVICE",
        2: "-dCompatibilityLevel",
        3: "-dPDFSETTINGS",  # /screen /ebook /printer /prepress
        4: "-dNOPAUSE",
        5: "-dQUIET",
        6: "-dBATCH",
        # 7: "-sOutputFile"  # handled separately at runtime
    }

    # ---------- Default options using shorthand codes ----------
    qpdf_defaults = {
        1: None,  # --warning-exit-0
        2: "compress",  # --stream-data=compress
        3: "generate",  # --object-streams=generate
        4: None,  # --optimize-images
        5: "5",  # --jpeg-quality=5
    }
    gs_defaults = {
        1: "pdfwrite",  # -sDEVICE=pdfwrite
        2: "1.3",  # -dCompatibilityLevel=1.3
        3: "/prepress",  # -dPDFSETTINGS=/screen|/ebook|/printer|/prepress
        4: None,  # -dNOPAUSE
        5: None,  # -dQUIET
        6: None,  # -dBATCH
    }

    # ---------- Helpers ----------
    def _as_path(p):
        return Path(p) if not isinstance(p, Path) else p

    def _run(cmd, title):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.stdout.strip():
            print(f"[{title}] STDOUT:\n{res.stdout.strip()}")
        if res.stderr.strip():
            print(f"[{title}] STDERR:\n{res.stderr.strip()}")
        if res.returncode != 0:
            raise RuntimeError(f"{title} failed with code {res.returncode}")

    def _increment_suffix(path: Path) -> Path:
        # name.pdf -> name_1.pdf; name_2.pdf -> name_3.pdf
        m = re.search(r"(.*)_(\d+)$", path.stem)
        if m:
            base, n = m.group(1), int(m.group(2)) + 1
            return path.with_name(f"{base}_{n}{path.suffix}")
        else:
            return path.with_name(f"{path.stem}_1{path.suffix}")

    def _target_path(src: Path) -> Path:
        if wts == 1:
            return src
        base = (Path(output_dir) if output_dir else src.parent) / src.name
        out = _increment_suffix(base)
        while out.exists():
            out = _increment_suffix(out)
        return out

    def _normalize_overrides(overrides, opt_map):
        """
        Accept numeric or full-name keys, return a dict keyed by real names.
        Value False removes option. None is flag. Else becomes string.
        """
        if not overrides:
            return {}
        norm = {}
        for k, v in overrides.items():
            if isinstance(k, int):
                if k not in opt_map:
                    raise KeyError(f"Unknown shorthand key: {k}")
                name = opt_map[k]
            else:
                name = str(k)
            norm[name] = v
        return norm

    def _merge_opts(defaults, opt_map, overrides):
        """
        Convert shorthand defaults to a dict keyed by real names,
        then apply normalized overrides (which may delete with False).
        """
        merged = {opt_map[k]: v for k, v in defaults.items()}
        for name, val in overrides.items():
            if val is False:
                merged.pop(name, None)  # remove option
            else:
                # Normalize some values, e.g., PDFSETTINGS allow "ebook" -> "/ebook"
                if (
                    name == "-dPDFSETTINGS"
                    and isinstance(val, str)
                    and not val.startswith("/")
                ):
                    val = "/" + val
                merged[name] = val
        return merged

    def _build_cmd(exe, merged_opts, trailing_args=None):
        """
        Build command list: flags (None) -> just the name; valued -> name=value.
        Order follows gs_map/qpdf_map numerical order, then any extras.
        """
        cmd = [str(exe)]
        # Respect preferred order defined by the map; then append others deterministically.
        order = []
        # choose the correct map by probing a key
        sample = next(iter(merged_opts.keys()), None)
        current_map = (
            gs_map
            if sample and sample.startswith("-d") or sample == "-sDEVICE"
            else qpdf_map
        )

        for code in current_map:
            name = current_map[code]
            if name in merged_opts:
                val = merged_opts[name]
                if val is None:
                    cmd.append(name)
                else:
                    cmd.append(f"{name}={val}")
                order.append(name)
        # Append any options not present in the map (or extra custom ones)
        for name, val in merged_opts.items():
            if name in order:
                continue
            if val is None:
                cmd.append(name)
            else:
                cmd.append(f"{name}={val}")

        if trailing_args:
            cmd.extend(map(str, trailing_args))
        return cmd

    # ---------- Validate inputs ----------
    if not isinstance(pdf_paths, (list, tuple)) or not pdf_paths:
        raise ValueError("pdf_paths must be a non-empty list")
    pdf_paths = [_as_path(p) for p in pdf_paths]
    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {p}")

    qpdf_exec = _as_path(qpdf_path)
    gs_exec = _as_path(gs_path)
    if not qpdf_exec.exists():
        raise FileNotFoundError(f"QPDF not found: {qpdf_exec}")
    if not gs_exec.exists():
        raise FileNotFoundError(f"Ghostscript not found: {gs_exec}")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Merge options ----------
    qpdf_over = _normalize_overrides(qpdf_opts, qpdf_map)
    gs_over = _normalize_overrides(gs_opts, gs_map)

    qpdf_merged = _merge_opts(qpdf_defaults, qpdf_map, qpdf_over)
    gs_merged = _merge_opts(gs_defaults, gs_map, gs_over)

    results = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_rs_"))

    try:
        for src in pdf_paths:
            qpdf_out = tmp_dir / f"{src.stem}_qpdf.pdf"
            target = _target_path(src)

            # -------- QPDF stage --------
            qpdf_cmd = _build_cmd(
                qpdf_exec, qpdf_merged, trailing_args=[str(src), str(qpdf_out)]
            )
            _run(qpdf_cmd, f"QPDF: {src.name}")

            # -------- Ghostscript stage --------
            if wts == 1:
                tmp_final = src.with_name(src.stem + ".__tmp__.pdf")
                out_arg = "-sOutputFile=" + str(tmp_final)
            else:
                out_arg = "-sOutputFile=" + str(target)

            # Build GS command: options first, then add output and input
            gs_cmd = _build_cmd(
                gs_exec, gs_merged, trailing_args=[out_arg, str(qpdf_out)]
            )
            _run(gs_cmd, f"Ghostscript: {src.name}")

            if wts == 1:
                tmp_final.replace(src)
                target = src

            if not keep_temp:
                qpdf_out.unlink(missing_ok=True)

            results.append(target.resolve())

        return results

    finally:
        if not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------
# Example usage:
outputs = pdf_rs(
    [r"C:\5Startemplib\Arhat\test\pdf_rs\0.pdf"],
    wts=0,
    qpdf_opts={5: "10"},  # --jpeg-quality=10
    gs_opts={
        3: "/prepress",
        2: "1.3",
    },  # -dPDFSETTINGS=/prepress, -dCompatibilityLevel=1.3
    keep_temp=0,
)
print(outputs)
