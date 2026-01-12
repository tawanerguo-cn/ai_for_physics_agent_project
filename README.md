# AI for Physics Agent Project (MVP)

一个最小可用的“本地物理计算工具 + LLM 调度”项目：  
LLM 负责把自然语言解析为结构化参数，通过 **OpenAI-compatible API（默认 DeepSeek）** 触发本地工具执行；  
本地工具在 **Julia/Python 本机**计算并输出 `PNG + JSON 数据` 到 `out/` 目录。

当前已内置工具（全部支持 JSON stdin → JSON stdout 协议）：

- `ssh_band`：SSH 模型（1D）  
  - `boundary="pbc"`：能带 \(E(k)\)  
  - `boundary="obc"`：有限链谱（可选 `overlay_pbc=true` 叠加 PBC 对比）
- `kitaev_band`：Kitaev chain（1D p-wave 超导，BdG）  
  - `boundary="pbc"`：BdG 能带（\(\pm E(k)\)）  
  - `boundary="obc"`：有限链 BdG 谱（支持 `overlay_pbc=true`）
- `aah_harper`：Aubry–André / Harper（AAH，1D 准周期势/Harper）  
  - `boundary="obc"`：有限链谱 + IPR（局域化指标，可视化为点大小）  
  - `boundary="pbc"`：
    - `pbc_mode="bloch"`：当 \(\beta=p/q\)（周期势）时输出 **q 条 Bloch 能带** \(E_n(k)\)  
    - `pbc_mode="ring"`：一般 \(\beta\) 下做有限环（周期跃迁）谱 + IPR
- `clean_out`：清理 `out/` 目录生成文件（默认非交互，适合 agent workflow）

本仓库强调：
- **本地计算**（数值计算不交给云端）
- **统一工具协议**（stdin JSON → stdout JSON，产物落盘）
- **Windows 友好**（处理 PowerShell、编码、python alias 等常见坑）
- **可扩展**（工具注册表模式：新增模型只需加 executor + schema）

---

## 仓库结构（关键文件）

~~~
ai_for_physics_agent_project/
  agent_ssh_runner.py            # LLM 调度器：自然语言 → tool call → 本地执行
  Project.toml / Manifest.toml   # Julia 环境（建议提交以锁依赖）
  tools/
    ssh_band_tool.jl             # SSH：PBC/OBC/overlay
    kitaev_band_tool.jl          # Kitaev chain：PBC/OBC/overlay
    aa_harper_tool.jl            # AAH：OBC，PBC(bloch/ring)
    clean_out.py                 # 清理 out：stdin JSON → stdout JSON
  out/
    .gitkeep                     # 保留目录结构（输出不建议提交）
~~~

---

## 环境与依赖

### 必需
- Julia（建议 1.10+，以你本地为准）
- Python（用于 runner 与 clean_out）
- PowerShell（或同等 shell）

### LLM API（OpenAI-compatible）
默认配置为 DeepSeek（可替换为任何 OpenAI-compatible 服务）：

- 必需环境变量（择一即可）：
  - `DEEPSEEK_API_KEY`（推荐）
  - 或 `OPENAI_API_KEY`
- 可选环境变量：
  - `OPENAI_BASE_URL`（默认 `https://api.deepseek.com`）
  - `OPENAI_MODEL`（默认 `deepseek-chat`）
  - `AGENT_DEBUG_TOOL_CALLS=1`（打印 tool call 与返回，便于调试）
  - `JULIA="C:\path\to\julia.exe"`（若 julia 不在 PATH）

---

## 快速开始

### 0) 克隆与进入目录

~~~powershell
git clone https://github.com/tawanerguo-cn/ai_for_physics_agent_project.git
cd ai_for_physics_agent_project
~~~

---

## 1) Julia 工具单测（强烈推荐先做）

所有 Julia 工具均支持：`stdin JSON -> stdout JSON`，并在 `out/` 生成 `PNG + data.json`。

### 1.1 SSH（PBC / OBC + overlay）

**PBC 能带：**

~~~powershell
'{"boundary":"pbc","p":0.6,"b":0,"n_k":401,"out_dir":"out","prefix":"case_ssh_pbc_p06","overwrite":true}' |
  julia --project=. .\tools\ssh_band_tool.jl
~~~

**OBC 有限链谱（带 overlay 对比 PBC）：**

~~~powershell
'{"boundary":"obc","p":0.6,"b":0,"N":80,"overlay_pbc":true,"n_k":401,"out_dir":"out","prefix":"case_ssh_obc_p06","overwrite":true}' |
  julia --project=. .\tools\ssh_band_tool.jl
~~~

---

### 1.2 Kitaev chain（PBC / OBC + overlay）

**PBC BdG 能带：**

~~~powershell
'{"boundary":"pbc","t":1.0,"mu":0.5,"delta":1.0,"n_k":401,"out_dir":"out","prefix":"case_kitaev_pbc","overwrite":true}' |
  julia --project=. .\tools\kitaev_band_tool.jl
~~~

**OBC 有限链谱（带 overlay 对比 PBC）：**

~~~powershell
'{"boundary":"obc","t":1.0,"mu":0.5,"delta":1.0,"N":80,"overlay_pbc":true,"n_k":401,"out_dir":"out","prefix":"case_kitaev_obc","overwrite":true}' |
  julia --project=. .\tools\kitaev_band_tool.jl
~~~

---

### 1.3 AAH（OBC）与（PBC Bloch / PBC ring）

**OBC（准周期势，\(\lambda > 2t\) 期望更局域，IPR 更大）：**

~~~powershell
'{"boundary":"obc","t":1.0,"lambda":2.5,"beta":0.61803398875,"phi":0.0,"N":200,"out_dir":"out","prefix":"case_aah_obc","overwrite":true}' |
  julia --project=. .\tools\aa_harper_tool.jl
~~~

**PBC Bloch bands（\(\beta=p/q\)，示例 \(\beta=1/3\) 会有 3 条能带）：**

~~~powershell
'{"boundary":"pbc","pbc_mode":"bloch","t":1.0,"lambda":1.0,"beta_num":1,"beta_den":3,"phi":0.0,"n_k":401,"out_dir":"out","prefix":"case_aah_pbc_bloch_1_3","overwrite":true}' |
  julia --project=. .\tools\aa_harper_tool.jl
~~~

**PBC ring（有限环谱 + IPR，适合一般 \(\beta\)）：**

~~~powershell
'{"boundary":"pbc","pbc_mode":"ring","t":1.0,"lambda":2.5,"beta":0.61803398875,"phi":0.0,"N":200,"out_dir":"out","prefix":"case_aah_pbc_ring","overwrite":true}' |
  julia --project=. .\tools\aa_harper_tool.jl
~~~

---

## 2) 清理工具单测（clean_out）

**dry-run（不删，只列出）：**

~~~powershell
'{"out_dir":"out","dry_run":true}' | python .\tools\clean_out.py
~~~

**执行删除：**

~~~powershell
'{"out_dir":"out","dry_run":false}' | python .\tools\clean_out.py
~~~

---

## 3) 运行 agent（LLM 调度本地工具）

### 3.1 设置 API Key

~~~powershell
$env:DEEPSEEK_API_KEY="你的key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_MODEL="deepseek-chat"
$env:AGENT_DEBUG_TOOL_CALLS="1"
~~~

可选（若 julia 不在 PATH）：

~~~powershell
$env:JULIA="C:\path\to\julia.exe"
~~~

---

### 3.2 自然语言示例

**清理 out：**

~~~powershell
python .\agent_ssh_runner.py "清理out"
~~~

**SSH：OBC + overlay：**

~~~powershell
python .\agent_ssh_runner.py "先清理out，然后用OBC画SSH：p=0.6, N=80，并且overlay对比PBC（n_k=401），prefix=demo_ssh_obc_p06"
~~~

**Kitaev：OBC + overlay（拓扑相示例）：**

~~~powershell
python .\agent_ssh_runner.py "先清理out，然后画Kitaev链：OBC，N=80，t=1，mu=0.5，delta=1，要求对比PBC(overlay)，n_k=401，prefix=demo_kitaev_obc"
~~~

**AAH：一次会话内多轮工具调用（先 OBC 再 PBC Bloch bands）：**

~~~powershell
python .\agent_ssh_runner.py "先清理out目录，然后连续做两次 AAH(Aubry–André/Harper) 计算并分别出图、写数据：(1) OBC：t=1.0，lambda=2.5，beta=0.61803398875，phi=0，N=200，prefix=aah_test_obc，overwrite=true；(2) PBC Bloch 能带：t=1.0，lambda=1.0，beta=1/3（解析为 beta_num=1,beta_den=3），phi=0，n_k=401，pbc_mode=bloch，prefix=aah_test_pbc_bloch，overwrite=true。"
~~~

---

## 工具协议（Tool I/O Contract）

所有工具统一：
- **stdin**：单个 JSON object
- **stdout**：最终必须输出一个可解析的 JSON object（允许前面有日志，但最后必须有 `{...}`）
- **落盘产物**：PNG 图 + data JSON（路径在 stdout JSON 里返回）

stdout JSON 典型结构：

~~~json
{
  "ok": true,
  "model": "...",
  "boundary": "pbc",
  "params": { "...": "..." },
  "png": "out/xxx.png",
  "data_json": "out/xxx_data.json",
  "extra_pngs": []
}
~~~

---

## 各工具参数速查

### 1) `ssh_band`
- 必需：`p`
- 常用：
  - `boundary`: `"pbc"` / `"obc"`
  - `b`: staggered onsite（默认 0）
  - `n_k`: PBC k 点数（OBC overlay 时也用）
  - `N`: OBC 有限链长度（unit cells）
  - `overlay_pbc`: OBC 时是否同时输出 PBC 对比
  - `out_dir`, `prefix`, `overwrite`

### 2) `kitaev_band`
- 必需：`mu`
- 常用：
  - `boundary`: `"pbc"` / `"obc"`
  - `t`: hopping（默认 1）
  - `delta`: pairing（默认 1）
  - `n_k`, `N`, `overlay_pbc`
  - `out_dir`, `prefix`, `overwrite`

### 3) `aah_harper`
- 必需：`lambda`
- 常用：
  - `boundary`: `"obc"` / `"pbc"`
  - `t`, `phi`
  - `beta`（float，准周期）或 `beta_num/beta_den`（有理数 p/q）
  - `pbc_mode`（仅 PBC）：
    - `"bloch"`：需要 `beta_num/beta_den`（输出 q 条能带）
    - `"ring"`：用 `beta` + `N`（输出有限环谱 + IPR）
  - `n_k`（bloch bands），`N`（finite chain/ring）
  - `out_dir`, `prefix`, `overwrite`

---

## 常见问题（Windows）

### 1) PowerShell 里 `python` 打开 Microsoft Store
这是 Windows 的 App execution aliases 劫持导致的。建议：
- 用明确 python 路径运行（以你本地为准），或
- Windows 设置中关闭：  
  Settings → Apps → Advanced app settings → App execution aliases → 关闭 `python.exe / python3.exe`

### 2) Python 依赖缺失（如 `ModuleNotFoundError: openai`）
确保你运行 runner 的那个 python 环境已安装 `openai` SDK：

~~~powershell
C:\path\to\python.exe -m pip install openai
~~~

### 3) 输出乱码
优先检查：
- PowerShell 编码设置
- 工具脚本是否打印了非 UTF-8 日志

---

## 开发与扩展

`agent_ssh_runner.py` 采用“工具注册表”模式：  
新增模型/工具只需两步：

1. 实现一个 `executor(args) -> dict`（把自然语言参数映射到工具 stdin JSON）
2. 在 `TOOL_SPECS` 注册 tool schema 与 executor

后续可扩展方向：
- 更多 tight-binding / BdG 模型（Creutz ladder、QWZ strip、Haldane ribbon 等）
- 输出更丰富指标（Zak phase、Chern number、edge localization 等）
- 复合 workflow（clean → run → summarize → run again）
