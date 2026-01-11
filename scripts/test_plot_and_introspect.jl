# scripts/test_plot_and_introspect.jl
using Pkg

println("== Julia environment ==")
println("Active project: ", Base.active_project())
Pkg.status()

println("\n== Import packages ==")
using TopologicalNumbers
using LatticeQM
using PythonPlot  # TopologicalNumbers 依赖里已有 PythonPlot（用于画图并保存）

println("\n== 1) Plot sanity check via TopologicalNumbers (SSH model) ==")

# 取自 TopologicalNumbers README：SSH Hamiltonian + showBand
function H0_SSH(k, p)
    t1 = 1.0
    t2 = p
    return [
        0.0               (t1 + t2*exp(-im*k))
        (t1 + t2*exp(im*k))     0.0
    ]
end

H(k) = H0_SSH(k, 1.1)

# 输出目录
outdir = joinpath(@__DIR__, "..", "out")
mkpath(outdir)

# showBand 默认会绘图（PythonPlot 后端），我们先直接调用一次
# value=false/disp=true 的用法同 README 示例
showBand(H; value=false, disp=true)

# 确保保存一张图（即使 showBand 的内部保存策略变化，我们也显式保存一次）
PythonPlot.savefig(joinpath(outdir, "ssh_band_topologicalnumbers.png"), dpi=200)
println("Saved: out/ssh_band_topologicalnumbers.png")

println("\n== 2) LatticeQM import OK. Now introspect likely high-value APIs ==")

# 列出 LatticeQM 里可能与你任务相关的符号（band / floquet / response / mean-field / operator 等关键词）
function pick_symbols(mod; keywords::Vector{String})
    syms = names(mod; all=true)
    hits = Symbol[]
    for s in syms
        ss = lowercase(String(s))
        for kw in keywords
            if occursin(kw, ss)
                push!(hits, s)
                break
            end
        end
    end
    return sort(unique(hits))
end

keywords = ["band", "disp", "spectrum", "eigen", "operator", "hop", "lattice",
            "berry", "chern", "floquet", "mean", "response", "kubo", "green", "dos"]

hits = pick_symbols(LatticeQM; keywords=keywords)
println("Candidate symbols in LatticeQM = ")
println(hits)

# 对“确实是函数”的候选符号打印 methods 签名，帮助你下一步快速定位怎么调用
for s in hits
    try
        obj = getfield(LatticeQM, s)
        if obj isa Function
            println("\n---- methods(LatticeQM.$s) ----")
            show(stdout, MIME("text/plain"), methods(obj))
            println()
        end
    catch err
        # 有些名字可能是宏/类型/内部 binding，不影响
    end
end

println("\nAll done.")
