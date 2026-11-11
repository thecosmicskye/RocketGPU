#!/usr/bin/env python3


import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import time

SOURCE_EXT = (".c", ".cc", ".cpp", ".cxx", ".cu")
HEADER_EXT = (".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".cuh", ".def")
ENV_KEYS = ("CC", "CXX", "CUDAHOSTCXX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CPATH",
            "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_INC_PATH", "SDKROOT",
            "MACOSX_DEPLOYMENT_TARGET", "LANG", "LC_ALL")
RESPONSE_OPTS = ("--options-file", "-optf")
FORCED_INCLUDE_OPTS = ("-include", "-imacros", "--pre-include", "-include-pch")
INCLUDE_OPTS = ("-I", "-isystem", "-iquote", "-idirafter", "--include-path", "--system-include")


class Uncached(Exception):
    pass


def file_hash(h, label, path):
    h.update(("\0F " + label + "\0").encode())
    with open(path, "rb") as f:
        h.update(f.read())


def is_build_dir(path):
    return os.path.isfile(os.path.join(path, "CMakeCache.txt"))


def tree_hash(h, label, root, skip):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d != ".git" and os.path.realpath(os.path.join(dirpath, d)) != skip
                             and not is_build_dir(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            if name.lower().endswith(SOURCE_EXT) or name.endswith((".o", ".obj", ".a", ".so", ".pyc")) or \
                    name.startswith("._") or name == ".DS_Store":
                continue
            path = os.path.join(dirpath, name)
            file_hash(h, label + "/" + os.path.relpath(path, root).replace(os.sep, "/"), path)


def stat_hash(h, label, root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            st = os.stat(os.path.join(dirpath, name))
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            h.update(("\0S %s/%s %d %d" % (label, rel, st.st_size, st.st_mtime_ns)).encode())


def expand(args, depth=0):
    if depth > 8:
        raise Uncached("response files nested too deep")
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        path = None
        if a.startswith("@") and len(a) > 1:
            path = a[1:]
        elif a in RESPONSE_OPTS and i + 1 < len(args):
            path = args[i + 1]
            i += 1
        elif any(a.startswith(o + "=") for o in RESPONSE_OPTS):
            path = a.split("=", 1)[1]
        if path is None:
            out.append(a)
            i += 1
            continue
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            raise Uncached("cannot read response file %s" % path)
        out.extend(expand(shlex.split(text), depth + 1))
        i += 1
    return out


def inside(path, base):
    return path == base or path.startswith(base + os.sep)


SPELLINGS = []


def relative(text):
    for path, mark in SPELLINGS:
        text = text.replace(path, mark)
    return text


def unmapped(deptext, root, build):
    for tok in deptext.replace("\\\n", " ").split():
        if tok.startswith("/") and (inside(os.path.realpath(tok), root) or inside(os.path.realpath(tok), build)):
            return True
    return False


def main():
    argv = sys.argv[1:]
    cut = argv.index("--")
    opts, cmd = argv[:cut], argv[cut + 1 :]
    root = os.path.realpath(opts[opts.index("--root") + 1])
    cache = os.path.realpath(opts[opts.index("--cache") + 1])
    build = os.path.realpath(os.getcwd())
    global SPELLINGS
    names = {(build, "@B@"), (os.getcwd(), "@B@"), (root, "@R@"), (os.path.abspath(opts[opts.index("--root") + 1]), "@R@")}
    pwd = os.environ.get("PWD", "")
    if pwd and os.path.realpath(pwd) == build:
        names.add((pwd, "@B@"))
    SPELLINGS = sorted(names, key=lambda n: (-len(n[0]), n[1]))
    compiler = cmd[0]
    if os.environ.get("OBJ_CACHE_OFF") or "-c" not in cmd[1:]:
        sys.exit(subprocess.call(cmd))
    try:
        return cached(cmd, root, cache, build)
    except Uncached as e:
        sys.stderr.write("obj_cache: uncached (%s)\n" % e)
        return subprocess.call(cmd)


def tool_version(exe):
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise Uncached("cannot run %s --version" % exe)
    if r.returncode != 0:
        raise Uncached("%s --version failed" % exe)
    return r.stdout


def cached(cmd, root, cache, build):
    compiler = cmd[0]
    h = hashlib.sha256()
    args = expand(cmd[1:])
    source = output = depfile = None
    include_dirs = []
    forced = []
    kept = []
    parsed = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "-MF", "-MT", "-MQ") and i + 1 < len(args):
            if a == "-o":
                output = args[i + 1]
            elif a == "-MF":
                depfile = args[i + 1]
            i += 2
            continue
        if a in ("-MD", "-MMD"):
            i += 1
            continue
        if a in INCLUDE_OPTS + FORCED_INCLUDE_OPTS and i + 1 < len(args):
            (forced if a in FORCED_INCLUDE_OPTS else include_dirs).append(args[i + 1])
            parsed += [a, args[i + 1]]
            i += 2
            continue
        joined = next((o for o in INCLUDE_OPTS + FORCED_INCLUDE_OPTS if a.startswith(o) and len(a) > len(o)), None)
        if joined is not None:
            value = a[len(joined):].lstrip("=")
            (forced if joined in FORCED_INCLUDE_OPTS else include_dirs).append(value)
            parsed.append(a)
            i += 1
            continue
        if not a.startswith("-") and a.lower().endswith(SOURCE_EXT) and os.path.isfile(a):
            if source is not None:
                source = False
            else:
                source = a
                i += 1
                continue
        kept.append(a)
        i += 1

    if not source or not output:
        raise Uncached("not one source and one output")
    for a in kept:
        if any(o in a for o in ("-include", "-imacros", "-optf", "--options-file", "-Xclang", "@")):
            raise Uncached("argument it cannot key: %s" % a)
    kept += parsed

    h.update(tool_version(compiler))
    if "nvcc" in os.path.basename(compiler):
        host = None
        for k, a in enumerate(args):
            if a in ("-ccbin", "--compiler-bindir") and k + 1 < len(args):
                host = args[k + 1]
            elif a.startswith(("-ccbin=", "--compiler-bindir=")):
                host = a.split("=", 1)[1]
        host = host or os.environ.get("NVCC_CCBIN") or os.environ.get("CUDAHOSTCXX") or "c++"
        if os.path.isdir(host):
            host = os.path.join(host, "c++")
        h.update(b"\0H")
        h.update(tool_version(host))
    for k in ENV_KEYS:
        h.update(("\0E %s=%s" % (k, os.environ.get(k, ""))).encode())
    for a in kept:
        h.update(("\0A " + relative(a)).encode())
    file_hash(h, "source", source)
    src_dir = os.path.dirname(os.path.realpath(source))
    for name in sorted(os.listdir(src_dir)):
        if os.path.isfile(os.path.join(src_dir, name)) and not name.lower().endswith(SOURCE_EXT):
            file_hash(h, "beside/" + name, os.path.join(src_dir, name))
    tree_hash(h, "@R@", root, cache)
    for f in forced:
        real = os.path.realpath(f)
        if not os.path.isfile(real):
            raise Uncached("forced include %s not found" % f)
        file_hash(h, "forced/" + relative(real), real)
    for d in include_dirs:
        real = os.path.realpath(d)
        if not os.path.isdir(real):
            continue
        if inside(cache, real) or inside(real, cache):
            raise Uncached("include directory overlaps the cache")
        if inside(real, root) and not inside(real, build):
            continue
        if inside(real, build):
            tree_hash(h, relative(real), real, cache)
        else:
            stat_hash(h, real, real)
    key = h.hexdigest()
    stored = os.path.join(cache, key[:2], key + ".o")
    name = os.path.basename(source)

    stored_dep = stored[:-2] + ".d"
    if os.path.isfile(stored) and (not depfile or os.path.isfile(stored_dep)):
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        shutil.copyfile(stored, output)
        if depfile:
            with open(stored_dep) as f:
                text = f.read()
            with open(depfile, "w") as f:
                f.write(text.replace("@OUT@", output).replace("@B@", build).replace("@R@", root))
        sys.stderr.write("obj_cache: hit %s\n" % name)
        return 0

    t0 = time.time()
    rc = subprocess.call(cmd)
    text = None
    if rc == 0 and depfile and os.path.isfile(depfile):
        with open(depfile) as f:
            text = relative(f.read().replace(output, "@OUT@"))
        if unmapped(text, root, build):
            sys.stderr.write("obj_cache: not stored %s (dependency file names a project path it cannot map)\n" % name)
            return rc
    if rc == 0 and os.path.isfile(output) and (not depfile or text is not None):
        os.makedirs(os.path.dirname(stored), exist_ok=True)
        if depfile:
            tmp = "%s.%d.tmp" % (stored_dep, os.getpid())
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, stored_dep)
        tmp = "%s.%d.tmp" % (stored, os.getpid())
        shutil.copyfile(output, tmp)
        os.replace(tmp, stored)
    sys.stderr.write("obj_cache: miss %s (%.1f s)\n" % (name, time.time() - t0))
    return rc


if __name__ == "__main__":
    sys.exit(main())
