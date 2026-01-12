#!/usr/bin/env julia
# tools/aa_harper_tool.jl
#
# Aubry–André / Harper (AAH) model tool.
#
# Model:
#   H = -t Σ_i (|i><i+1| + |i+1><i|) + λ Σ_i cos(2πβ i + φ) |i><i|
#
# Boundaries:
#   - OBC: finite chain (no wrap)
#   - PBC:
#       * pbc_mode="bloch" : require beta_num/beta_den = p/q (periodic potential) -> q Bloch bands E_n(k)
#       * pbc_mode="ring"  : finite ring (wrap hopping) -> spectrum scatter with IPR
#
# Input JSON (stdin):
#   {
#     "boundary": "pbc" | "obc",
#     "pbc_mode": "bloch" | "ring",      # only for boundary="pbc"; optional
#     "t": 1.0,                          # default 1.0
#     "lambda": 1.0,                     # potential strength λ (required)
#     "beta": 0.618...,                  # optional if beta_num/beta_den given
#     "beta_num": 1, "beta_den": 3,      # optional; for bloch bands require these
#     "phi": 0.0,                        # default 0.0
#     "n_k": 401,                        # for bloch bands
#     "N": 200,                          # for OBC, and for pbc_mode="ring"
#     "out_dir": "out",
#     "prefix": "demo",
#     "overwrite": true | false
#   }
#
# Output JSON (stdout):
#   {
#     "ok": true,
#     "model": "aubry_andre_harper",
#     "boundary": "...",
#     "mode": "...",
#     "params": {...},
#     "png": "...",
#     "data_json": "...",
#     "extra_pngs": [...]
#   }

ENV["MPLBACKEND"] = "Agg"  # set before importing PythonPlot

using JSON3
using LinearAlgebra
using PythonPlot
using Printf

# ------------------------
# Utilities (same style as your existing tools)
# ------------------------

strip_bom(s::AbstractString) = startswith(s, "\ufeff") ? s[nextind(s, firstindex(s)):end] : s

function read_stdin_all()::String
    return read(stdin, String)
end

function extract_json_object(raw::String)::String
    s = strip(strip_bom(raw))
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
# Core: AAH Hamiltonians
# ------------------------

# onsite potential with i=1..N (index shift absorbed into phi)
@inline function onsite(i::Int, lambda::Float64, beta::Float64, phi::Float64)
    return lambda * cos(2π * beta * i + phi)
end

function aah_hamiltonian_obc(t::Float64, lambda::Float64, beta::Float64, phi::Float64, N::Int)
    H = zeros(Float64, N, N)
    @inbounds for i in 1:N
        H[i, i] = onsite(i, lambda, beta, phi)
        if i < N
            H[i, i+1] = -t
            H[i+1, i] = -t
        end
    end
    return Symmetric(H)
end

function aah_hamiltonian_ring(t::Float64, lambda::Float64, beta::Float64, phi::Float64, N::Int)
    H = Matrix{Float64}(aah_hamiltonian_obc(t, lambda, beta, phi, N))
    # periodic hopping
    H[1, N] = -t
    H[N, 1] = -t
    return Symmetric(H)
end

function spectrum_ipr(H::Symmetric{Float64, Matrix{Float64}})
    F = eigen(H)
    evals = Vector{Float64}(F.values)
    vecs = F.vectors
    ipr = zeros(Float64, length(evals))
    @inbounds for j in 1:length(evals)
        ψ = view(vecs, :, j)
        # IPR = Σ |ψ_i|^4  (normalized ψ)
        s = 0.0
        for i in 1:length(ψ)
            s += (ψ[i]^2)^2
        end
        ipr[j] = s
    end
    return evals, ipr
end

# Bloch bands for beta = p/q (period-q potential)
function aah_bloch_bands(t::Float64, lambda::Float64, p::Int, q::Int, phi::Float64, n_k::Int)
    ks = range(-pi, pi; length=n_k)
    bands = zeros(Float64, q, n_k)

    # precompute onsite for m=1..q
    V = [onsite(m, lambda, p / q, phi) for m in 1:q]

    Hk = zeros(ComplexF64, q, q)
    @inbounds for (ik, k) in enumerate(ks)
        fill!(Hk, 0.0 + 0.0im)
        for m in 1:q
            Hk[m, m] = V[m]
            if m < q
                Hk[m, m+1] = -t
                Hk[m+1, m] = -t
            end
        end
        # boundary hopping with Bloch phase
        Hk[1, q] = -t * exp(-1im * k)
        Hk[q, 1] = -t * exp(+1im * k)

        vals = eigvals(Hermitian(Hk))
        # ensure ascending order
        for n in 1:q
            bands[n, ik] = real(vals[n])
        end
    end
    return ks, bands
end

# ------------------------
# Plotting
# ------------------------

function plot_bands(ks, bands::Matrix{Float64}, out_png::String; title::String="")
    fig = PythonPlot.figure()
    ax = fig.add_subplot(1, 1, 1)
    nb, nk = size(bands)
    for n in 1:nb
        ax.plot(ks, view(bands, n, :))
    end
    ax.set_xlabel("k")
    ax.set_ylabel("E")
    if !isempty(strip(title))
        ax.set_title(title)
    end
    fig.tight_layout()
    fig.savefig(out_png; dpi=200)
    PythonPlot.close(fig)
end

function plot_spectrum_scatter(evals::Vector{Float64}, ipr::Vector{Float64}, out_png::String; title::String="")
    fig = PythonPlot.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = collect(1:length(evals))

    # size scaled by normalized IPR
    mn = minimum(ipr)
    mx = maximum(ipr)
    denom = (mx - mn)
    w = denom > 0 ? (ipr .- mn) ./ denom : fill(0.0, length(ipr))
    sizes = 10 .+ 200 .* clamp.(w, 0.0, 1.0)

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

    boundary = lowercase(get_string(obj, ["boundary", "bc"], "obc"))
    if !(boundary in ("pbc", "obc"))
        error("Invalid boundary='$boundary'. Expected 'pbc' or 'obc'.")
    end

    t = get_float(obj, ["t", "hop", "hopping"]; required=false, default=1.0, name="t")
    lambda = get_float(obj, ["lambda", "λ", "V", "potential"]; required=true, name="lambda")
    phi = get_float(obj, ["phi", "ϕ", "phase"]; required=false, default=0.0, name="phi")

    out_dir = ensure_out_dir(get_string(obj, ["out_dir", "outdir", "output_dir"], "out"))
    overwrite = get_bool(obj, ["overwrite", "force"], false)
    prefix = sanitize_prefix(get_string(obj, ["prefix", "name", "tag"], "demo"))

    N = get_int(obj, ["N", "n", "L", "size"], 200)
    n_k = get_int(obj, ["n_k", "nk"], 401)

    beta_num = get_int(obj, ["beta_num", "p"], 0)
    beta_den = get_int(obj, ["beta_den", "q"], 0)
    beta = 0.0
    beta_has_rational = (beta_num > 0 && beta_den > 0)

    if beta_has_rational
        beta = beta_num / beta_den
    else
        beta = get_float(obj, ["beta", "β"], required=true, name="beta")
    end

    extra_pngs = String[]
    data = Dict{String, Any}()

    # heuristic: for incommensurate beta, localization transition at lambda = 2|t|
    loc_heur = abs(lambda) < 2.0 * abs(t) ? "extended (heuristic)" :
               abs(lambda) > 2.0 * abs(t) ? "localized (heuristic)" : "critical (heuristic)"

    if boundary == "obc"
        H = aah_hamiltonian_obc(t, lambda, beta, phi, max(N, 2))
        evals, ipr = spectrum_ipr(H)

        main_png = choose_output_path(out_dir, "$(prefix)_aah_obc_spectrum.png"; overwrite=overwrite)
        data_json = choose_output_path(out_dir, "$(prefix)_aah_obc_data.json"; overwrite=overwrite)

        title = @sprintf("AAH spectrum (OBC): t=%.3g, λ=%.3g, β=%.6g, ϕ=%.3g, N=%d (%s)",
                         t, lambda, beta, phi, N, loc_heur)
        plot_spectrum_scatter(evals, ipr, main_png; title=title)

        data["model"] = "aubry_andre_harper"
        data["boundary"] = "obc"
        data["mode"] = "finite_chain"
        data["t"] = t
        data["lambda"] = lambda
        data["beta"] = beta
        data["phi"] = phi
        data["N"] = N
        data["localization_heuristic"] = loc_heur
        data["eigenvalues"] = evals
        data["ipr"] = ipr

        open(data_json, "w") do io
            JSON3.pretty(io, data)
        end

        out = Dict(
            "ok" => true,
            "model" => "aubry_andre_harper",
            "boundary" => "obc",
            "mode" => "finite_chain",
            "params" => Dict("t" => t, "lambda" => lambda, "beta" => beta, "phi" => phi, "N" => N, "localization_heuristic" => loc_heur),
            "png" => main_png,
            "data_json" => data_json,
            "extra_pngs" => extra_pngs,
        )
        print(JSON3.write(out))
        return
    end

    # boundary == "pbc"
    pbc_mode = lowercase(get_string(obj, ["pbc_mode", "mode"], ""))
    if isempty(pbc_mode)
        pbc_mode = beta_has_rational ? "bloch" : "ring"
    end
    if !(pbc_mode in ("bloch", "ring"))
        error("Invalid pbc_mode='$pbc_mode'. Expected 'bloch' or 'ring'.")
    end

    if pbc_mode == "bloch"
        if !beta_has_rational
            error("pbc_mode='bloch' requires beta_num and beta_den (beta=p/q).")
        end
        q = beta_den
        p = mod(beta_num, q)
        if p == 0
            # allow p=q, but in that case beta integer -> trivial; still valid
            p = beta_num
        end

        ks, bands = aah_bloch_bands(t, lambda, beta_num, beta_den, phi, max(n_k, 5))

        main_png = choose_output_path(out_dir, "$(prefix)_aah_pbc_bloch_bands.png"; overwrite=overwrite)
        data_json = choose_output_path(out_dir, "$(prefix)_aah_pbc_bloch_data.json"; overwrite=overwrite)

        title = @sprintf("AAH Bloch bands (PBC, β=%d/%d): t=%.3g, λ=%.3g, ϕ=%.3g, n_k=%d",
                         beta_num, beta_den, t, lambda, phi, n_k)
        plot_bands(ks, bands, main_png; title=title)

        data["model"] = "aubry_andre_harper"
        data["boundary"] = "pbc"
        data["mode"] = "bloch_bands"
        data["t"] = t
        data["lambda"] = lambda
        data["beta_num"] = beta_num
        data["beta_den"] = beta_den
        data["beta"] = beta
        data["phi"] = phi
        data["n_k"] = n_k
        data["k"] = collect(ks)
        data["bands"] = [collect(view(bands, n, :)) for n in 1:size(bands, 1)]

        open(data_json, "w") do io
            JSON3.pretty(io, data)
        end

        out = Dict(
            "ok" => true,
            "model" => "aubry_andre_harper",
            "boundary" => "pbc",
            "mode" => "bloch_bands",
            "params" => Dict("t" => t, "lambda" => lambda, "beta_num" => beta_num, "beta_den" => beta_den, "phi" => phi, "n_k" => n_k),
            "png" => main_png,
            "data_json" => data_json,
            "extra_pngs" => extra_pngs,
        )
        print(JSON3.write(out))
        return
    end

    # pbc_mode == "ring"
    H = aah_hamiltonian_ring(t, lambda, beta, phi, max(N, 2))
    evals, ipr = spectrum_ipr(H)

    main_png = choose_output_path(out_dir, "$(prefix)_aah_pbc_ring_spectrum.png"; overwrite=overwrite)
    data_json = choose_output_path(out_dir, "$(prefix)_aah_pbc_ring_data.json"; overwrite=overwrite)

    title = @sprintf("AAH spectrum (PBC ring): t=%.3g, λ=%.3g, β=%.6g, ϕ=%.3g, N=%d (%s)",
                     t, lambda, beta, phi, N, loc_heur)
    plot_spectrum_scatter(evals, ipr, main_png; title=title)

    data["model"] = "aubry_andre_harper"
    data["boundary"] = "pbc"
    data["mode"] = "finite_ring"
    data["t"] = t
    data["lambda"] = lambda
    data["beta"] = beta
    data["phi"] = phi
    data["N"] = N
    data["localization_heuristic"] = loc_heur
    data["eigenvalues"] = evals
    data["ipr"] = ipr

    open(data_json, "w") do io
        JSON3.pretty(io, data)
    end

    out = Dict(
        "ok" => true,
        "model" => "aubry_andre_harper",
        "boundary" => "pbc",
        "mode" => "finite_ring",
        "params" => Dict("t" => t, "lambda" => lambda, "beta" => beta, "phi" => phi, "N" => N, "localization_heuristic" => loc_heur),
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
    println(stderr, "[aa_harper_tool] ERROR: ", msg)
    out = Dict("ok" => false, "error" => msg)
    print(JSON3.write(out))
end
