# scripts/show_tree.jl
root = pwd()

# 你可以按需改这两个参数，避免输出太长
const MAX_DEPTH = 4
const SKIP_DIRS = Set([
    ".git", ".julia", "artifacts", "node_modules", ".venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "build", "dist"
])

function depth_from_root(rel::AbstractString)
    rel == "." && return 0
    return length(splitpath(rel))
end

function print_tree(io::IO, root::AbstractString; max_depth::Int=MAX_DEPTH)
    println(io, "PWD = ", root)
    println(io, "---- tree (depth <= ", max_depth, ") ----")

    for (dir, dirs, files) in walkdir(root; topdown=true)
        rel = relpath(dir, root)
        d = depth_from_root(rel)
        d > max_depth && continue

        # 过滤掉不需要看的目录
        filter!(x -> !(x in SKIP_DIRS), dirs)

        indent = repeat("  ", d)
        println(io, indent, "[D] ", (rel == "." ? "." : rel))

        # 文件太多时你也可以加过滤；这里先全打
        for f in sort(files)
            println(io, indent, "  - ", f)
        end
    end
end

# 同时写到文件，便于你整段复制给我
out_path = joinpath(root, "agent_project_tree.txt")
open(out_path, "w") do io
    print_tree(io, root; max_depth=MAX_DEPTH)
end

# 控制台也打印一份（方便你确认）
print_tree(stdout, root; max_depth=MAX_DEPTH)

println("\n[Saved] ", out_path)
