#!/usr/bin/env python3


import os
import re
import sys

MODULES = {
    "step": ["StepKernel"],
    "post": [
        "PreTickKernel", "StagedVehicleIndependenceKernel", "StagedCarPreKernel", "StagedCarPostKernel", "StagedPadPreKernel",
        "StagedEnvTailKernel", "StagedPostFusedKernel", "StagedPostPreFusedKernel",
    ],
    "world": [
        "StagedWorldKernel<kWorldStepContacts>", "StagedWorldKernel<kWorldStepContactsCarsApart>",
        "StagedWorldKernel<kWorldStepContacts, kWorld1Lanes, false, 1>",
        "StagedWorldKernel<kWorldStepContactsCarsApart, 1, false, 2>", "StagedBallWorldKernel", "StagedW3ProxiesKernel",
    ],
    "rows": [
        "StagedWorldKernel", "StagedCarWorldRowsKernel", "StagedCarWorldRowsSplitKernel", "StagedCarWorldRowsListKernel",
        "StagedCarWorldRowsLightKernel", "StagedCarWorldRowsSplitPlanesKernel",
    ],
    "rays": [
        "StageWheelRaysKernel", "StageWheelRaysLanesKernel", "StageWheelRaysWarpKernel", "StagedPredictiveRaysKernel",
        "PrefilterCarWorldMeshCandidatesKernel", "StageCarWorldMeshCandidatesKernel", "StageCarWorldGatherKernel",
        "CarWorldTriangleHitsKernel",
    ],
    "misc": [
        "PrepareRespawnSlotsKernel", "ResetToKickoffKernel", "ApplyEnvCommandsKernel", "CanonicalizeImportedCarsKernel",
        "CanonicalizeImportedBallsKernel", "ZeroInt32Kernel", "InterleaveGroupsKernel", "AwakeFirstGroupsKernel",
        "IotaInt32Kernel", "ZeroTickRowsKernel",
    ],
}
SHARED = {"WriteModuleDeviceSymbol"}

DEVICE_MARKS = re.compile(r"\b(__device__|__constant__|__host__|ROCKETGPU_DEVICE|ROCKETGPU_HD|ROCKETGPU_DEVICE_GLOBAL)\b")
KERNEL_MARK = re.compile(r"\b(ROCKETGPU_GLOBAL|__global__)\b")
TYPE_START = re.compile(r"^(template\s*<.*>\s*)?(struct|class|union|enum|using|typedef|constexpr|static\s+constexpr)\b",
                        re.S)


def strip_code(line, state):
    out = []
    i = 0
    n = len(line)
    while i < n:
        if state["block"]:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out)
            state["block"] = False
            i = j + 2
            continue
        c = line[i]
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            state["block"] = True
            i += 2
            continue
        if c in "\"'":
            j = i + 1
            while j < n and line[j] != c:
                j += 2 if line[j] == "\\" else 1
            out.append(" ")
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_blocks(lines):
    items = []
    cur, code = [], []
    depth = 0
    paren = 0
    state = {"block": False}
    pp = None
    for line in lines:
        stripped = line.strip()
        if pp is not None:
            pp.append(line)
            if not stripped.endswith("\\"):
                items.append(("pp", pp))
                pp = None
            continue
        if depth == 0 and paren == 0 and not code and not state["block"] and stripped.startswith("#"):
            if cur:
                items.append(("pp", cur))
                cur = []
            if stripped.endswith("\\"):
                pp = [line]
            else:
                items.append(("pp", [line]))
            continue
        c = strip_code(line, state)
        cur.append(line)
        if c.strip():
            code.append(c)
        for ch in c:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "(":
                paren += 1
            elif ch == ")":
                paren -= 1
        if depth == 0 and paren == 0 and code:
            tail = code[-1].rstrip()
            if tail.endswith("}") or tail.endswith("};") or tail.endswith(";"):
                items.append(("block", cur, " ".join(code)))
                cur, code = [], []
    if cur:
        items.append(("pp", cur))
    return items


def header_of(code):
    depth = 0
    for i, ch in enumerate(code):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "{;":
            return code[:i]
    return code


def func_name(head):
    for m in re.finditer(r"(\w+)\s*\(", head):
        if m.group(1) not in ("__launch_bounds__", "__align__", "alignas", "decltype"):
            return m.group(1)
    return None


def classify(code):
    head = header_of(code)
    if KERNEL_MARK.search(head):
        return "kernel", func_name(head)
    name = func_name(head)
    if name in SHARED:
        return "shared", name
    if DEVICE_MARKS.search(head) or TYPE_START.match(head.strip()):
        if re.match(r"^(template\s*<.*>\s*)?(struct|class)\b", head.strip(), re.S) and not DEVICE_MARKS.search(code):
            return "host", name
        return "shared", name
    return "host", name


INTERNAL = re.compile(r"\b(static|ROCKETGPU_SPLIT_STATIC|inline|__forceinline__|constexpr|extern|ROCKETGPU_HD|ROCKETGPU_DEVICE|"
                      r"ROCKETGPU_HOST_INLINE|ROCKETGPU_FORCEINLINE)\b")


def external_definition(code):
    head = header_of(code).strip()
    if re.match(r"^template\s*<", head) or TYPE_START.match(head) or INTERNAL.search(head):
        return None
    if re.match(r"^(struct|class|union|enum|using|typedef|namespace)\b", head):
        return None
    after = code[len(header_of(code)) :].lstrip()
    if "(" in head.split("=")[0] and after.startswith("{"):
        return "function"
    if "(" not in head and re.match(r"^(\w+\s+)*const\b", head) is None:
        return "variable"
    return None


def declared_name(code):
    head = header_of(code).strip()
    m = re.match(r"^(?:template\s*<.*?>\s*)?(?:struct|class|union|enum(?:\s+(?:class|struct))?)\s+(\w+)", head, re.S)
    if m:
        return m.group(1)
    m = re.match(r"^(?:template\s*<.*?>\s*)?using\s+(\w+)\s*=", head, re.S)
    if m:
        return m.group(1)
    if re.match(r"^(static\s+)?constexpr\b", head):
        left = head.split("=")[0]
        if "(" in left:
            return func_name(left)
        names = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", left))
        return names[-1] if names else None
    return None


def as_static(block_lines):
    out = list(block_lines)
    state = {"block": False}
    for i, line in enumerate(out):
        if strip_code(line, state).strip():
            indent = len(line) - len(line.lstrip())
            out[i] = line[:indent] + "static " + line[indent:]
            return out
    return out


def kernel_decl(block_lines):
    text = "".join(block_lines)
    state = {"block": False}
    depth = 0
    pos = 0
    for line in block_lines:
        c = strip_code(line, state)
        if len(c) < len(line):
            c = c + " " * (len(line) - len(c))
        for i, ch in enumerate(c):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "{" and depth == 0:
                sig = text[: pos + i].rstrip()
                lines = sig.split("\n")
                while lines and (lines[0].strip().startswith("//") or not lines[0].strip()):
                    lines.pop(0)
                return "\n".join(lines) + ";\n"
        pos += len(line)
    raise ValueError("no kernel body")


def mask_code(text):
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif text[i] in "\"'":
            j = i + 1
            while j < n and text[j] != text[i] and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j += 1
        else:
            i += 1
            continue
        for k in range(i, min(j, n)):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def match_close(code, i, open_ch, close_ch):
    depth = 0
    for j in range(i, len(code)):
        if code[j] == open_ch:
            depth += 1
        elif code[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced %s" % open_ch)


def split_top(text):
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [" ".join(p.split()) for p in parts]


def kernel_params(decl, name):
    m = re.search(r"\b%s\s*\(" % re.escape(name), decl)
    start = m.end() - 1
    end = match_close(decl, start, "(", ")")
    params = []
    for p in split_top(decl[start + 1 : end - 1]):
        arg = re.findall(r"\w+", p.split("=")[0])[-1]
        if "__grid_constant__" in p:
            p = "const " + re.sub(r"\b(const|__grid_constant__)\b", "", p[: p.rindex(arg)]).strip() + "& " + arg
        params.append((p, arg))
    return params


class SplitError(Exception):
    pass


def rewrite_host(host, templates, assign):

    code = mask_code(host)
    out, pos = [], 0
    instances = {}
    pattern = re.compile(r"\b(%s)\b" % "|".join(re.escape(n) for n in templates))
    for m in pattern.finditer(code):
        name = m.group(1)
        d = host.rfind("#line ", 0, m.start())
        line = int(host[d + 6 : host.index(" ", d + 6)]) + host.count("\n", d, m.start()) - 1
        j = m.end()
        while code[j].isspace():
            j += 1
        if code[j] != "<":
            if code[m.start() - 1 : m.start()] == " " and re.search(r"ROCKETGPU_GLOBAL\s+void\s+$", code[: m.start()]):
                continue
            raise SplitError("%s used without template arguments (line %d)" % (name, line))
        k = match_close(code, j, "<", ">")
        args = " ".join(code[j + 1 : k - 1].split())
        module = assign(name, args)
        if module is None:
            continue
        key = (name, args)
        inst = instances.setdefault(key, "%s_%d" % (name, len([1 for n, _ in instances if n == name])))
        a = k
        while code[a].isspace():
            a += 1
        before = code[: m.start()].rstrip()
        if code.startswith("<<<", a):
            c = code.index(">>>", a)
            cfg = split_top(host[a + 3 : c])
            if not 2 <= len(cfg) <= 4:
                raise SplitError("%s<%s>: launch config %r (line %d)" % (name, args, cfg, line))
            cfg += ["0", "0"][: 4 - len(cfg)]
            p = c + 3
            while code[p].isspace():
                p += 1
            if code[p] != "(":
                raise SplitError("%s<%s>: no argument list after <<<>>> (line %d)" % (name, args, line))
            out.append(host[pos : m.start()])
            out.append("SplitLaunch_%s(%s, " % (inst, ", ".join(cfg)))
            pos = p + 1
        elif before.endswith("&"):
            out.append(host[pos : len(before) - 1])
            out.append("SplitKernel_%s()" % inst)
            pos = k
        else:
            raise SplitError("%s<%s>: neither launched with <<<>>> nor taken with & (line %d)" % (name, args, line))
    out.append(host[pos:])
    return "".join(out), instances


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    with open(src) as f:
        lines = f.readlines()
    rel = os.path.relpath(src, out_dir).replace("\\", "/")
    open_idx = next(i for i, l in enumerate(lines) if l.strip() == "namespace detail {")
    close_idx = max(i for i, l in enumerate(lines) if l.strip() == "}" and i > open_idx)
    close_idx -= 1
    owner = {}
    for module, entries in MODULES.items():
        for e in entries:
            owner[" ".join(e.split()).replace("< ", "<").replace(" >", ">")] = module
    items = parse_blocks(lines[open_idx + 1 : close_idx])
    placed = []
    line_no = open_idx + 2
    for item in items:
        placed.append((line_no, item))
        line_no += len(item[1])

    def assign(name, args):
        return owner.get("%s<%s>" % (name, args), owner.get(name))

    templates = {}
    for start, item in placed:
        if item[0] == "block":
            cls, name = classify(item[2])
            if cls == "kernel" and kernel_decl(item[1]).lstrip().startswith("template"):
                templates[name] = (start, item[1])

    device_code = []
    plain = {}
    for start, item in placed:
        if item[0] != "block":
            continue
        cls, name = classify(item[2])
        if cls == "kernel":
            if name not in templates or any(k == name or k.startswith(name + "<") for k in owner) or name in owner:
                device_code.append(item[2])
        elif cls == "shared":
            declared = None if DEVICE_MARKS.search(item[2]) or name in SHARED else declared_name(item[2])
            if declared is None:
                device_code.append(item[2])
            else:
                plain[start] = (declared, item[2])
    whole = mask_code("".join(lines))
    for m in re.finditer(r"\b(%s)\s*<" % "|".join(re.escape(n) for n in templates), whole) if templates else []:
        try:
            device_code.append(whole[m.end() - 1 : match_close(whole, m.end() - 1, "<", ">")])
        except ValueError:
            pass
    host_only = set(plain)
    grew = True
    while grew:
        grew = False
        text = " ".join(device_code)
        for start in sorted(host_only):
            declared, code = plain[start]
            if re.search(r"\b%s\b" % re.escape(declared), text):
                host_only.discard(start)
                device_code.append(code)
                grew = True

    host_text = []
    module_text = {m: [] for m in MODULES}

    def emit(dst, start, block_lines):
        if dst is host_text:
            dst.append('#line %d "%s"\n' % (start, rel))
        dst.extend(block_lines)

    for start, item in placed:
        if item[0] == "pp":
            for dst in [host_text] + list(module_text.values()):
                emit(dst, start, item[1])
            continue
        _, block_lines, code = item
        cls, name = classify(code)
        if cls == "kernel" and name in templates:
            owned = any(k == name or k.startswith(name + "<") for k in owner)
            for dst in ([] if owned else [host_text]) + list(module_text.values()):
                emit(dst, start, block_lines)
            host_text.append("// split_cuda thunks: %s\n" % name)
        elif cls == "kernel" and name in owner:
            emit(module_text[owner[name]], start, block_lines)
            emit(host_text, start, [kernel_decl(block_lines)])
        elif cls in ("kernel", "host"):
            emit(host_text, start, block_lines)
        else:
            ext = external_definition(code)
            if ext == "variable":
                name = re.findall(r"\w+", header_of(code).split("=")[0])[-1]
                if re.search(r"\b(__device__|__constant__)\b", code):
                    emit(host_text, start, block_lines)
                    continue
            if ext == "function" and re.search(r"::\s*~?\w+\s*\($", header_of(code).split("(")[0] + "("):
                sys.exit("split_cuda.py: %s line %d: out-of-class definition of %s is needed by every module; define "
                         "it inline" % (src, start, name))
            if ext == "variable" or (ext and name in SHARED):
                sys.exit("split_cuda.py: %s line %d: %s %s has external linkage and is needed by every module; make it "
                         "static, inline or const" % (src, start, ext, name))
            emit(host_text, start, block_lines)
            for dst in module_text.values() if start not in host_only else []:
                emit(dst, start, as_static(block_lines) if ext else block_lines)
    try:
        host_body, instances = rewrite_host("".join(host_text), {n: d for n, (_, d) in templates.items()}, assign)
    except SplitError as e:
        sys.exit("split_cuda.py: %s: %s" % (src, e))
    thunk_decls = {n: [] for n in templates}
    thunks = {m: [] for m in MODULES}
    for (name, args), inst in instances.items():
        _, block_lines = templates[name]
        params = kernel_params(kernel_decl(block_lines), name)
        sig_launch = "void SplitLaunch_%s(dim3 grid, dim3 block, size_t smem, cudaStream_t stream, %s)" % (
            inst, ", ".join(p for p, _ in params))
        sig_ptr = "const void* SplitKernel_%s()" % inst
        thunk_decls[name].append("%s;\n%s;\n" % (sig_launch, sig_ptr))
        thunks[assign(name, args)].append(
            "%s {\n    %s<%s><<<grid, block, smem, stream>>>(%s);\n}\n%s {\n    return reinterpret_cast<const void*>(&%s<%s>);\n}\n"
            % (sig_launch, name, args, ", ".join(a for _, a in params), sig_ptr, name, args))
    for name, decls in thunk_decls.items():
        host_body = host_body.replace("// split_cuda thunks: %s\n" % name, "".join(decls), 1)

    prefix = "".join(lines[: open_idx + 1])
    split_tu = "#define ROCKETGPU_CUDA_SPLIT_TU 1\n"
    for m in MODULES:
        writer = ("cudaError_t WriteDeviceSymbol_%s(DeviceSymbol symbol, const void* src, size_t bytes, size_t offset,\n"
                  "                              cudaStream_t stream, bool async) {\n"
                  "    return WriteModuleDeviceSymbol(symbol, src, bytes, offset, stream, async);\n}\n" % m)
        body = "".join(module_text[m]) + "\n" + "".join(thunks[m]) + writer
        write(os.path.join(out_dir, "rocketgpu_cuda_%s.cu" % m), split_tu + prefix + body + "".join(lines[close_idx:]))
    defs = "#define ROCKETGPU_CUDA_MODULES(X) %s\n" % " ".join("X(%s)" % m for m in MODULES)
    host_body = re.sub(r'(?m)^#line \d+ "[^"\n]*"\n', "", host_body)
    write(os.path.join(out_dir, "rocketgpu_cuda_host.cu"), split_tu + defs + prefix + host_body + "".join(lines[close_idx:]))


def write(path, text):
    try:
        with open(path) as f:
            if f.read() == text:
                return
    except OSError:
        pass
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
