```markdown
# AI for Physics Agent Project (MVP)

一个最小可用的“本地物理计算工具 + LLM 调度”项目：  
LLM 负责把自然语言解析为结构化参数，通过 **OpenAI-compatible API（当前默认 DeepSeek）** 触发本地工具执行；  
本地工具目前包括：

- `ssh_band`：用 Julia 在本地计算 SSH 模型 **PBC 能带** 或 **OBC 有限链谱（可叠加 overlay PBC 对比）**，输出 `PNG + JSON 数据`
- `clean_out`：清理 `out/` 目录下的生成文件（非交互，适用于 agent 工具调用）

本仓库强调：
- **本地计算**（不把计算交给云端）
- **工具输入输出统一为 JSON（stdin→stdout）**，便于扩展更多物理模型/更多工具
- **Windows 友好**（处理 PowerShell、编码、Python alias 等常见坑）

---

## 仓库结构（关键文件）

```

agent_project/
agent_ssh_runner.py           # LLM 调度器：解析自然语言 → tool call → 本地执行
Project.toml / Manifest.toml  # Julia 环境（建议提交以锁依赖版本）
tools/
ssh_band_tool.jl            # SSH 工具：PBC/OBC/overlay，stdin JSON → stdout JSON
clean_out.py                # 清理工具：stdin JSON → stdout JSON（非交互）
clean_out.cmd               # 可选：手工清理脚本（若保留）
scripts/
test_plot_and_introspect.jl # 早期测试脚本
show_tree.jl                # 输出目录树（调试用）
out/
.gitkeep                    # 保留目录结构，实际输出不建议提交

````

---

## 功能概览

### 1) SSH 模型（`ssh_band`）
- `boundary="pbc"`：计算布里渊区 `k ∈ [-π, π]` 上的两条能带
- `boundary="obc"`：构造有限长度链（`N` 个 unit cell），直接对角化得到谱；可选 `overlay_pbc=true` 输出 PBC 对比

输出：
- PNG 图像（能带或谱）
- JSON 数据（数值数组/参数元信息）
- stdout 返回一个 JSON：包含 `ok=true/false`、参数回显、输出文件路径等

### 2) 清理输出（`clean_out`）
- 删除 `out/` 下除 `.gitkeep` 外的文件/目录
- 支持 `dry_run=true` 仅列出将删除的目标
- **默认非交互**（stdin 有 JSON payload 时自动执行），适合 agent 调用

---

## 环境与依赖

### 必需
- Windows PowerShell（或同等 shell）
- Julia（建议 1.10+；以你本地为准）
- Python（用于 runner 和 clean_out）

### LLM API（OpenAI-compatible）
- 默认：DeepSeek（通过 `base_url=https://api.deepseek.com`）
- 需要设置环境变量：
  - `DEEPSEEK_API_KEY`（或 `OPENAI_API_KEY`）

---

## 快速开始

### 0) 克隆与进入目录
```powershell
git clone https://github.com/tawanerguo-cn/ai_for_physics_agent_project.git
cd ai_for_physics_agent_project
````

> 若你的仓库目录名仍是 `agent_project/`，以实际目录为准。

### 1) Julia 单测（推荐先做）

确保你在仓库根目录（含 `Project.toml`）：

**PBC：**

```powershell
'{"boundary":"pbc","p":0.6,"b":0,"n_k":401,"out_dir":"out","prefix":"case_pbc_p06","overwrite":false}' |
  julia --project=. .\tools\ssh_band_tool.jl
```

**OBC（带 overlay）：**

```powershell
'{"boundary":"obc","p":0.6,"b":0,"N":80,"overlay_pbc":true,"n_k":401,"out_dir":"out","prefix":"case_obc_p06","overwrite":false}' |
  julia --project=. .\tools\ssh_band_tool.jl
```

两条都应返回类似：

```json
{"ok":true, ...}
```

并在 `out/` 下生成对应的 PNG 与数据 JSON。

### 2) 清理工具单测

**dry-run（不删，只列出）：**

```powershell
'{"out_dir":"out","dry_run":true}' | python .\tools\clean_out.py
```

**执行删除：**

```powershell
'{"out_dir":"out","dry_run":false}' | python .\tools\clean_out.py
```

> 如果你的系统 `python` 会触发 Microsoft Store（Windows “App execution aliases” 问题），请改用明确的 Python 路径，见「常见问题」。

### 3) 运行 agent（LLM 调度本地工具）

#### 设置 API Key

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

可选（若 `julia` 未加入 PATH）：

```powershell
$env:JULIA="C:\path\to\julia.exe"
```

可选（打开 tool call 调试输出）：

```powershell
$env:AGENT_DEBUG_TOOL_CALLS="1"
```

#### 示例：清理 out

```powershell
python .\agent_ssh_runner.py "清理out"
```

#### 示例：OBC + overlay（对比 PBC）

```powershell
python .\agent_ssh_runner.py "先清理out，然后用OBC画SSH：p=0.6, N=80，并且overlay对比PBC（n_k=401），prefix=case_obc_p06"
```

---

## 工具协议（Tool I/O Contract）

### `tools/ssh_band_tool.jl`

* stdin：JSON object
* stdout：以 JSON object 结尾（允许前面有日志，但最后必须有可解析的 `{...}`）
* 关键参数（最常用）：

  * `boundary`: `"pbc"` 或 `"obc"`（默认 `"pbc"`）
  * `p`: number（SSH inter-cell hopping ratio，t1 固定 1.0）
  * `b`: number（staggered onsite term，默认 0）
  * `n_k`: int（PBC k 点数；OBC overlay 时也使用）
  * `N`: int（OBC 有限链长度：unit cells）
  * `overlay_pbc`: bool（OBC 时是否同时输出 PBC 对比）
  * `out_dir`: string（默认 `"out"`）
  * `prefix`: string（建议指定，避免混淆输出）
  * `overwrite`: bool（默认 false）

### `tools/clean_out.py`

* stdin：JSON object
* stdout：JSON object
* 关键参数：

  * `out_dir`: string（默认 `"out"`）
  * `dry_run`: bool（默认 false）
  * `keep_gitkeep`: bool（默认 true）

---

## 常见问题（Windows）

### 1) PowerShell 里 `python` 打开 Microsoft Store

这是 Windows 的 App execution aliases 劫持导致的。解决办法之一：

* 用明确的 python 路径运行，例如（以你本地实际路径为准）：

```powershell
.\.CondaPkg\.pixi\envs\default\python.exe .\agent_ssh_runner.py "清理out"
```

或在 Windows 设置中关闭：
**Settings → Apps → Advanced app settings → App execution aliases**
关闭 `python.exe / python3.exe`。

### 2) Python 依赖缺失（如 `ModuleNotFoundError: openai`）

如果你在某个 Python 环境里运行 runner，需要确保该环境已安装 `openai` SDK。

例如使用某个 python.exe：

```powershell
C:\path\to\python.exe -m pip install openai
```

> 建议固定一个环境，并在 README/脚本里明确调用该 python。

### 3) 输出编码/乱码

runner 对 Windows 做了 UTF-8 bytes 捕获与替换解码处理；若你仍遇到乱码，优先检查：

* PowerShell 的编码设置
* 工具脚本是否输出了非 UTF-8 的日志

---

## 开发与扩展

当前 `agent_ssh_runner.py` 的设计是“工具注册表”模式：
新增工具只需：

1. 实现一个 `executor(args)->dict`
2. 在 `TOOL_SPECS` 注册 tool schema 与 executor
   无需改动主循环。

适合后续扩展：

* 更多 tight-binding 模型（Haldane、Kitaev chain、Chern number 等）
* 只出数据不出图
* 复合工作流（clean → run → summarize → 继续 run）


