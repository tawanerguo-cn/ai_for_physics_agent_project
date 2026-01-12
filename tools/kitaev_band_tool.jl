#!/usr/bin/env julia
# tools/kitaev_band_tool.jl
#
# Kitaev chain (spinless p-wave superconductor) band/spectrum tool.
#
# Usage (PowerShell):
#   Get-Content .\tools\input_kitaev.json | julia --project=. .\tools\kitaev_band_tool.jl
#
# Input (JSON via stdin):
#   {
#     "boundary": "pbc" | "obc",          # default "pbc"
#     "t": 1.0,                           # hopping (default 1.0)
#     "mu": 0.5,                          # chemical potential (required)
#     "delta": 1.0,                       # pairing amplitude Δ (default 1.0)
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
#     "model": "kitaev_chain",
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

# ------------------------
# Utilities (mirrors ssh_band_tool.jl style)
# ------------------------

strip_bom(s::AbstractString) = startswith(s, "\ufeff") ? s[nextind(s, firstindex(s)):end] : s

function read_stdin_all()::String
    return read(stdin, String)
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
    v === nothing && return default
    return String(v)
end

function get_bool(dict, keys::Vector{String}, default::Bool)
    v = get_any(dict, keys)
    v === nothing && return default
    if v isa Bool
        return v
    elseif v isa Integer
        return v != 0
    elseif v isa AbstractString
        s = lowercase(strip(String(v)))
        if s in ("1", "true", "yes", "y", "on")
            return true
        elseif s in ("0", "false", "no", "n", "off")
            return false
        end
    end
    return default
end

function get_int(dict, keys::Vector{String}, default::Int)
    v = get_any(dict, keys)
    v === nothing && return default
    if v isa Integer
        return Int(v)
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
    if v isa Real
        return Float64(v)
    elseif v isa AbstractString
        return parse(Float64, strip(String(v)))
    end
    error("Could not parse '$name' as float (value=$v).")
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

# ------------------------
# Kitaev chain: PBC (BdG bands)
# ------------------------

"""
Kitaev chain in k-space (spinless p-wave superconductor).
E(k) = ± sqrt( ξ(k)^2 + Δ(k)^2 )
where ξ(k) = -mu - 2 t cos k, and Δ(k) = 2 delta sin k  (real delta).
"""
function kitaev_bands_pbc(t::Float64, mu::Float64, delta::Float64, n_k::Int)
    ks = range(-pi, pi; length=n_k)
    xi = -mu .- 2.0 * t .* cos.(ks)
    dk = 2.0 * delta .* sin.(ks)
    Ek = sqrt.(xi.^2 .+ dk.^2)
    return ks, -Ek, Ek, xi, dk
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
# Kitaev chain: OBC (finite chain BdG spectrum)
# ------------------------

"""
Build BdG Hamiltonian for an N-site open Kitaev chain in Nambu basis:
  Ψ = (c_1, ..., c_N, c_1^†, ..., c_N^†)^T

BdG matrix:
  H = [A  Δ;
       Δ†  -Aᵀ]

with A(i,i) = -mu, A(i,i+1)=A(i+1,i)=-t
and pairing Δ(i,i+1)= +delta, Δ(i+1,i)= -delta (antisymmetric p-wave).
For real parameters, Δ† = Δᵀ = -Δ.
"""
function kitaev_hamiltonian_obc(t::Float64, mu::Float64, delta::Float64, N::Int)
    A = zeros(ComplexF64, N, N)
    D = zeros(ComplexF64, N, N)  # pairing Δ

    @inbounds for i in 1:N
        A[i, i] += -mu
        if i < N
            # hopping
            A[i, i+1] += -t
            A[i+1, i] += -t
            # p-wave pairing
            D[i, i+1] += delta
            D[i+1, i] += -delta
        end
    end

    # assemble BdG
    H = zeros(ComplexF64, 2N, 2N)
    H[1:N, 1:N] .= A
    H[1:N, N+1:2N] .= D
    H[N+1:2N, 1:N] .= adjoint(D)          # Δ†
    H[N+1:2N, N+1:2N] .= -transpose(A)     # -Aᵀ

    return Hermitian(H)
end

function obc_spectrum_and_edgeweight(t::Float64, mu::Float64, delta::Float64, N::Int)
    H = kitaev_hamiltonian_obc(t, mu, delta, N)
    F = eigen(H)  # Hermitian eigen; sorted eigenvalues

    evals = Vector{Float64}(F.values)
    evecs = F.vectors  # columns are eigenvectors

    # edge-weight: (site 1, site N) for both particle(u) and hole(v) components
    idxs = Int[1, N, N + 1, 2N]

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

    boundary = lowercase(get_string(obj, ["boundary", "bc", "bcond"], "pbc"))
    if !(boundary in ("pbc", "obc"))
        error("Invalid boundary='$boundary'. Expected 'pbc' or 'obc'.")
    end

    t = get_float(obj, ["t", "hop", "hopping", "J", "t_hop"]; required=false, default=1.0, name="t")
    mu = get_float(obj, ["mu", "μ", "chem_pot", "chemical_potential", "chemicalPotential"]; required=true, name="mu")
    delta = get_float(obj, ["delta", "Δ", "pairing", "d", "Delta"]; required=false, default=1.0, name="delta")

    out_dir = ensure_out_dir(get_string(obj, ["out_dir", "outdir", "output", "output_dir"], "out"))
    overwrite = get_bool(obj, ["overwrite", "force"], false)

    prefix = sanitize_prefix(get_string(obj, ["prefix", "name", "tag"], "demo"))

    n_k = get_int(obj, ["n_k", "nk", "nK", "n_k_points", "k_points"], 401)
    N = get_int(obj, ["N", "n", "L", "size"], 80)
    overlay_pbc = get_bool(obj, ["overlay_pbc", "overlay", "with_pbc", "compare_pbc"], false)

    extra_pngs = String[]
    data = Dict{String, Any}()

    # phase heuristic (for real Δ != 0): topological if |mu| < 2|t|
    phase = (abs(delta) > 0 && abs(mu) < 2.0 * abs(t)) ? "topological" : "trivial"

    if boundary == "pbc"
        ks, Eminus, Eplus, xi, dk = kitaev_bands_pbc(t, mu, delta, n_k)

        main_png = choose_output_path(out_dir, "$(prefix)_kitaev_band.png"; overwrite=overwrite)
        data_json = choose_output_path(out_dir, "$(prefix)_kitaev_band_data.json"; overwrite=overwrite)

        gap = minimum(Eplus)
        title = @sprintf("Kitaev chain BdG bands (PBC): t=%.3g, mu=%.3g, Δ=%.3g, n_k=%d, gap≈%.3g (%s)",
                         t, mu, delta, n_k, gap, phase)
        plot_pbc_bands(ks, Eminus, Eplus, main_png; title=title)

        data["model"] = "kitaev_chain"
        data["boundary"] = "pbc"
        data["t"] = t
        data["mu"] = mu
        data["delta"] = delta
        data["n_k"] = n_k
        data["phase_heuristic"] = phase
        data["gap_estimate"] = gap
        data["k"] = collect(ks)
        data["bands"] = Dict("Eminus" => collect(Eminus), "Eplus" => collect(Eplus))
        data["xi_k"] = collect(xi)
        data["delta_k"] = collect(dk)

        open(data_json, "w") do io
            JSON3.pretty(io, data)
        end

        out = Dict(
            "ok" => true,
            "model" => "kitaev_chain",
            "boundary" => "pbc",
            "params" => Dict("t" => t, "mu" => mu, "delta" => delta, "n_k" => n_k, "phase_heuristic" => phase),
            "png" => main_png,
            "data_json" => data_json,
            "extra_pngs" => extra_pngs,
        )
        print(JSON3.write(out))
        return
    end

    # OBC branch
    evals, edge_w = obc_spectrum_and_edgeweight(t, mu, delta, N)

    main_png = choose_output_path(out_dir, "$(prefix)_kitaev_obc_spectrum.png"; overwrite=overwrite)
    data_json = choose_output_path(out_dir, "$(prefix)_kitaev_obc_spectrum_data.json"; overwrite=overwrite)

    gap = minimum(abs.(evals))
    title = @sprintf("Kitaev chain spectrum (OBC): t=%.3g, mu=%.3g, Δ=%.3g, N=%d, min|E|≈%.3g (%s)",
                     t, mu, delta, N, gap, phase)
    plot_obc_spectrum(evals, edge_w, main_png; title=title)

    data["model"] = "kitaev_chain"
    data["boundary"] = "obc"
    data["t"] = t
    data["mu"] = mu
    data["delta"] = delta
    data["N"] = N
    data["phase_heuristic"] = phase
    data["min_abs_eigenvalue"] = gap
    data["eigenvalues"] = evals
    data["edge_weight"] = edge_w

    if overlay_pbc
        ks, Eminus, Eplus, xi, dk = kitaev_bands_pbc(t, mu, delta, n_k)
        band_png = choose_output_path(out_dir, "$(prefix)_kitaev_pbc_band.png"; overwrite=overwrite)
        band_json = choose_output_path(out_dir, "$(prefix)_kitaev_pbc_band_data.json"; overwrite=overwrite)

        gap_pbc = minimum(Eplus)
        band_title = @sprintf("Kitaev chain BdG bands (PBC overlay): t=%.3g, mu=%.3g, Δ=%.3g, n_k=%d, gap≈%.3g (%s)",
                              t, mu, delta, n_k, gap_pbc, phase)
        plot_pbc_bands(ks, Eminus, Eplus, band_png; title=band_title)
        push!(extra_pngs, band_png)

        band_data = Dict{String, Any}()
        band_data["model"] = "kitaev_chain"
        band_data["boundary"] = "pbc"
        band_data["t"] = t
        band_data["mu"] = mu
        band_data["delta"] = delta
        band_data["n_k"] = n_k
        band_data["phase_heuristic"] = phase
        band_data["gap_estimate"] = gap_pbc
        band_data["k"] = collect(ks)
        band_data["bands"] = Dict("Eminus" => collect(Eminus), "Eplus" => collect(Eplus))
        band_data["xi_k"] = collect(xi)
        band_data["delta_k"] = collect(dk)

        open(band_json, "w") do io
            JSON3.pretty(io, band_data)
        end

        data["overlay_pbc"] = Dict("png" => band_png, "data_json" => band_json)
    end

    open(data_json, "w") do io
        JSON3.pretty(io, data)
    end

    out = Dict(
        "ok" => true,
        "model" => "kitaev_chain",
        "boundary" => "obc",
        "params" => Dict("t" => t, "mu" => mu, "delta" => delta, "N" => N, "phase_heuristic" => phase),
        "png" => main_png,
        "data_json" => data_json,
        "extra_pngs" => extra_pngs,
    )
    print(JSON3.write(out))
end

try
    main()
catch err
    msg = sprint(showerror, err)
    println(stderr, "[kitaev_band_tool] ERROR: ", msg)
    out = Dict("ok" => false, "error" => msg)
    print(JSON3.write(out))
end
