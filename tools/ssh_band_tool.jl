#!/usr/bin/env julia
# tools/ssh_band_tool.jl
#
# Usage (PowerShell):
#   Get-Content .\tools\input_demo.json | julia --project=. .\tools\ssh_band_tool.jl
#
# Input (JSON via stdin):
#   {
#     "boundary": "pbc" | "obc",          # default "pbc"
#     "p": 1.2,                           # required (aliases accepted)
#     "b": 0.0,                           # optional onsite staggering term (aliases accepted)
#     "n_k": 401,                         # for pbc, or for overlay_pbc in obc
#     "N": 80,                            # for obc
#     "overlay_pbc": true | false,        # only meaningful for obc; if true, also outputs a pbc band png
#     "out_dir": "out",
#     "prefix": "demo",
#     "overwrite": true | false           # default false; if false, auto-suffix avoids overwriting
#   }
#
# Output (JSON via stdout, single object):
#   {
#     "ok": true,
#     "boundary": "pbc" | "obc",
#     "params": {...},
#     "png": "path/to/main.png",
#     "data_json": "path/to/data.json",
#     "extra_pngs": ["optional/extra.png", ...]
#   }

ENV["MPLBACKEND"] = "Agg"  # set before importing PythonPlot/matplotlib

using JSON3
using LinearAlgebra
using PythonPlot
using Printf
using Dates

# ------------------------
# Utilities
# ------------------------

strip_bom(s::AbstractString) = startswith(s, "\ufeff") ? s[nextind(s, firstindex(s)):end] : s

function read_stdin_all()::String
    data = read(stdin, String)
    return data
end

function extract_json_object(raw::String)::String
    s = strip(strip_bom(raw))
    # tolerate leading junk/newlines; grab the first {...} block
    i = findfirst(==('{'), s)
    j = findlast(==('}'), s)
    if i === nothing || j === nothing || j < i
        error("No JSON object found in stdin.")
    end
    return strip(s[i:j])
end

function get_any(dict, keys::Vector{String})
    for k in keys
        if haskey(dict, k)
            return dict[k]
        end
    end
    return nothing
end

function get_string(dict, keys::Vector{String}, default::String)
    v = get_any(dict, keys)
    if v === nothing
        return default
    end
    return String(v)
end

function get_bool(dict, keys::Vector{String}, default::Bool)
    v = get_any(dict, keys)
    if v === nothing
        return default
    end
    if v isa Bool
        return v
    end
    if v isa Number
        return v != 0
    end
    if v isa AbstractString
        t = lowercase(strip(String(v)))
        if t in ("true", "t", "1", "yes", "y", "on")
            return true
        elseif t in ("false", "f", "0", "no", "n", "off")
            return false
        end
    end
    return default
end

function get_int(dict, keys::Vector{String}, default::Int)
    v = get_any(dict, keys)
    if v === nothing
        return default
    end
    if v isa Integer
        return Int(v)
    elseif v isa AbstractFloat
        return Int(round(v))
    elseif v isa AbstractString
        return parse(Int, strip(String(v)))
    end
    return default
end

function get_float(dict, keys::Vector{String}; required::Bool=false, default::Float64=0.0, name::String="param")
    v = get_any(dict, keys)
    if v === nothing
        if required
            error("Missing required parameter: $name (accepted keys: $(join(keys, ", "))).")
        end
        return default
    end
    if v isa Number
        return Float64(v)
    elseif v isa AbstractString
        return parse(Float64, strip(String(v)))
    end
    error("Parameter $name has unsupported type: $(typeof(v)).")
end

function sanitize_prefix(s::String)::String
    t = strip(s)
    t = replace(t, r"\s+" => "_")
    t = replace(t, ":" => "-")
    t = replace(t, "/" => "_")
    t = replace(t, "\\" => "_")
    return t
end

function ensure_out_dir(out_dir::String)::String
    d = strip(out_dir)
    isempty(d) && (d = "out")
    mkpath(d)
    return d
end

function unique_path(path::String)::String
    # If path doesn't exist, return it; otherwise append _001, _002, ...
    if !isfile(path)
        return path
    end
    base, ext = splitext(path)
    for i in 1:999
        cand = @sprintf("%s_%03d%s", base, i, ext)
        if !isfile(cand)
            return cand
        end
    end
    error("Could not find a unique filename for: $path")
end

function choose_output_path(out_dir::String, filename::String; overwrite::Bool=false)::String
    p = joinpath(out_dir, filename)
    return overwrite ? p : unique_path(p)
end

function now_tag()::String
    # Windows-safe timestamp
    return Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
end

# ------------------------
# SSH Model: PBC (bands)
# ------------------------

function ssh_bands_pbc(p::Float64, b::Float64, n_k::Int)
    t1 = 1.0
    t2 = p
    ks = range(-pi, pi; length=n_k)

    # E(k) = ± sqrt( t1^2 + t2^2 + 2 t1 t2 cos k + b^2 )
    Ek = sqrt.(t1^2 .+ t2^2 .+ 2.0 * t1 * t2 .* cos.(ks) .+ b^2)
    Eplus = Ek
    Eminus = -Ek
    return ks, Eminus, Eplus
end

function plot_pbc_bands(ks, Eminus, Eplus, out_png::String; title::String="")
    fig = PythonPlot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, Eplus)
    ax.plot(ks, Eminus)
    ax.set_xlabel("k")
    ax.set_ylabel("E")
    if !isempty(strip(title))
        ax.set_title(title)
    end
    fig.tight_layout()
    fig.savefig(out_png; dpi=200)
    PythonPlot.close(fig)
end

# ------------------------
# SSH Model: OBC (finite chain spectrum)
# ------------------------

function ssh_hamiltonian_obc(p::Float64, b::Float64, N::Int)
    t1 = 1.0
    t2 = p
    dim = 2N
    H = zeros(ComplexF64, dim, dim)

    # indexing: A_i = 2i-1, B_i = 2i
    for i in 1:N
        a = 2i - 1
        bb = 2i

        # onsite staggering term: +b on A, -b on B
        H[a, a] += b
        H[bb, bb] += -b

        # intracell hopping t1: A_i <-> B_i
        H[a, bb] += t1
        H[bb, a] += t1

        # intercell hopping t2: B_i <-> A_{i+1}
        if i < N
            a_next = 2(i + 1) - 1
            H[bb, a_next] += t2
            H[a_next, bb] += t2
        end
    end

    return Hermitian(H)
end

function obc_spectrum_and_edgeweight(p::Float64, b::Float64, N::Int)
    H = ssh_hamiltonian_obc(p, b, N)
    F = eigen(H)  # Hermitian eigen; returns sorted eigenvalues

    evals = Vector{Float64}(F.values)
    evecs = F.vectors  # columns are eigenvectors

    # simple edge-weight: probability on first and last unit cell (4 sites)
    idxs = Int[]
    push!(idxs, 1); push!(idxs, 2)
    push!(idxs, 2N - 1); push!(idxs, 2N)
    edge_w = zeros(Float64, length(evals))

    @inbounds for j in 1:length(evals)
        v = view(evecs, :, j)
        w = 0.0
        for ii in idxs
            w += abs2(v[ii])
        end
        edge_w[j] = w
    end

    return evals, edge_w
end

function plot_obc_spectrum(evals, edge_w, out_png::String; title::String="")
    fig = PythonPlot.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = collect(1:length(evals))

    # marker size mildly scales with edge weight for visibility (no explicit colors)
    sizes = 10 .+ 200 .* clamp.(edge_w, 0.0, 1.0)
    ax.scatter(xs, evals; s=sizes)

    ax.set_xlabel("eigenstate index")
    ax.set_ylabel("E")
    if !isempty(strip(title))
        ax.set_title(title)
    end
    fig.tight_layout()
    fig.savefig(out_png; dpi=200)
    PythonPlot.close(fig)
end

# ------------------------
# Main
# ------------------------

function main()
    raw = read_stdin_all()
    json_str = extract_json_object(raw)
    obj = JSON3.read(json_str)

    # canonical keys have priority; aliases are only used if canonical missing
    boundary = lowercase(get_string(obj, ["boundary", "bc", "bcond"], "pbc"))
    if !(boundary in ("pbc", "obc"))
        error("Invalid boundary='$boundary'. Expected 'pbc' or 'obc'.")
    end

    p = get_float(obj, ["p", "p0", "p_0", "t2_over_t1", "t2/t1", "ratio"]; required=true, name="p")
    b = get_float(obj, ["b", "mass", "m", "delta", "stagger", "stag"]; required=false, default=0.0, name="b")

    out_dir = ensure_out_dir(get_string(obj, ["out_dir", "outdir", "output", "output_dir"], "out"))

    prefix_in = get_any(obj, ["prefix", "name", "tag"])
    prefix = if prefix_in === nothing
        # default: include boundary and key params + timestamp to avoid overwrite
        sanitize_prefix(@sprintf("ssh_%s_p%.6g_b%.6g_%s", boundary, p, b, now_tag()))
    else
        sanitize_prefix(String(prefix_in))
    end

    overwrite = get_bool(obj, ["overwrite", "force"], false)

    # parameters: PBC
    n_k = get_int(obj, ["n_k", "nk", "nK", "k_points", "kpoints"], 301)
    n_k = max(n_k, 5)

    # parameters: OBC
    N = get_int(obj, ["N", "n", "L", "length"], 80)
    N = max(N, 2)

    overlay_pbc = get_bool(obj, ["overlay_pbc", "overlay", "with_pbc", "compare_pbc"], false)

    # output containers
    extra_pngs = String[]

    # ------------------------
    # Branch by boundary
    # ------------------------
    data = Dict{String, Any}()

    if boundary == "pbc"
        ks, Eminus, Eplus = ssh_bands_pbc(p, b, n_k)

        main_png = choose_output_path(out_dir, "$(prefix)_ssh_band.png"; overwrite=overwrite)
        data_json = choose_output_path(out_dir, "$(prefix)_ssh_band_data.json"; overwrite=overwrite)

        title = @sprintf("SSH bands (PBC): p=t2/t1=%.3g, b=%.3g, n_k=%d", p, b, n_k)
        plot_pbc_bands(ks, Eminus, Eplus, main_png; title=title)

        data["boundary"] = "pbc"
        data["p"] = p
        data["b"] = b
        data["n_k"] = n_k
        data["k"] = collect(ks)
        data["bands"] = Dict("Eminus" => collect(Eminus), "Eplus" => collect(Eplus))

        open(data_json, "w") do io
            JSON3.pretty(io, data)
        end


        out = Dict(
            "ok" => true,
            "boundary" => "pbc",
            "params" => Dict("p" => p, "b" => b, "n_k" => n_k, "out_dir" => out_dir, "prefix" => prefix, "overwrite" => overwrite),
            "png" => main_png,
            "data_json" => data_json,
            "extra_pngs" => extra_pngs,
        )

        print(JSON3.write(out))
        return
    end

    # boundary == "obc"
    evals, edge_w = obc_spectrum_and_edgeweight(p, b, N)

    main_png = choose_output_path(out_dir, "$(prefix)_ssh_obc_spectrum.png"; overwrite=overwrite)
    data_json = choose_output_path(out_dir, "$(prefix)_ssh_obc_spectrum_data.json"; overwrite=overwrite)

    title = @sprintf("SSH spectrum (OBC): p=t2/t1=%.3g, b=%.3g, N=%d", p, b, N)
    plot_obc_spectrum(evals, edge_w, main_png; title=title)

    data["boundary"] = "obc"
    data["p"] = p
    data["b"] = b
    data["N"] = N
    data["eigenvalues"] = evals
    data["edge_weight"] = edge_w

    # Optional overlay: also output PBC band figure/data for comparison
    if overlay_pbc
        ks, Eminus, Eplus = ssh_bands_pbc(p, b, n_k)
        band_png = choose_output_path(out_dir, "$(prefix)_ssh_pbc_band.png"; overwrite=overwrite)
        band_json = choose_output_path(out_dir, "$(prefix)_ssh_pbc_band_data.json"; overwrite=overwrite)

        band_title = @sprintf("SSH bands (PBC overlay): p=t2/t1=%.3g, b=%.3g, n_k=%d", p, b, n_k)
        plot_pbc_bands(ks, Eminus, Eplus, band_png; title=band_title)
        push!(extra_pngs, band_png)

        band_data = Dict{String, Any}()
        band_data["boundary"] = "pbc"
        band_data["p"] = p
        band_data["b"] = b
        band_data["n_k"] = n_k
        band_data["k"] = collect(ks)
        band_data["bands"] = Dict("Eminus" => collect(Eminus), "Eplus" => collect(Eplus))

        open(band_json, "w") do io
            JSON3.pretty(io, band_data)
        end


        # reference additional outputs in main data json
        data["overlay_pbc"] = Dict(
            "png" => band_png,
            "data_json" => band_json,
            "n_k" => n_k,
        )
    end

    open(data_json, "w") do io
        JSON3.pretty(io, data)
    end


    out = Dict(
        "ok" => true,
        "boundary" => "obc",
        "params" => Dict(
            "p" => p, "b" => b, "N" => N,
            "overlay_pbc" => overlay_pbc, "n_k" => n_k,
            "out_dir" => out_dir, "prefix" => prefix, "overwrite" => overwrite
        ),
        "png" => main_png,
        "data_json" => data_json,
        "extra_pngs" => extra_pngs,
    )

    print(JSON3.write(out))
end

try
    main()
catch err
    # Keep stdout machine-readable on failure too (helps LLM/tooling)
    msg = sprint(showerror, err)
    println(stderr, "[ssh_band_tool] ERROR: ", msg)

    out = Dict(
        "ok" => false,
        "error" => msg,
    )
    print(JSON3.write(out))
    exit(1)
end
