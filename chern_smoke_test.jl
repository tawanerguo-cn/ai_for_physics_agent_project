using LinearAlgebra

# Qi–Wu–Zhang 2-band Chern insulator (square lattice)
# H(k) = sin(kx) σx + sin(ky) σy + (m + cos(kx) + cos(ky)) σz
function qwx_hamiltonian(kx, ky; m=-1.0)
    sx = [0 1; 1 0]
    sy = [0 -im; im 0]
    sz = [1 0; 0 -1]
    return sin(kx)*sx + sin(ky)*sy + (m + cos(kx) + cos(ky))*sz
end

# Get eigenvector of target band (1 = lower energy for Hermitian 2x2 here, since eigenvalues are sorted ascending)
function band_eigvec(H::AbstractMatrix; band::Int=1)
    F = eigen(Hermitian(H))
    return F.vectors[:, band]
end

# Fukui-Hatsugai-Suzuki discretized Chern number on Nk x Nk grid
function chern_fhs(Hk::Function; band::Int=1, Nk::Int=31)
    # k grid: [0, 2π)
    dk = 2π / Nk

    # Precompute eigenvectors u(k)
    u = Vector{Matrix{ComplexF64}}(undef, Nk)  # each is (dim x Nk) for fixed kx index
    dim = size(Hk(0.0, 0.0), 1)
    for ix in 1:Nk
        u[ix] = Matrix{ComplexF64}(undef, dim, Nk)
        kx = (ix-1) * dk
        for iy in 1:Nk
            ky = (iy-1) * dk
            u[ix][:, iy] = band_eigvec(Hk(kx, ky); band=band)
        end
    end

    # Link variables Ux, Uy
    Ux = Matrix{ComplexF64}(undef, Nk, Nk)
    Uy = Matrix{ComplexF64}(undef, Nk, Nk)

    for ix in 1:Nk, iy in 1:Nk
        ixp = (ix % Nk) + 1
        iyp = (iy % Nk) + 1

        u00 = view(u[ix], :, iy)
        u10 = view(u[ixp], :, iy)
        u01 = view(u[ix], :, iyp)

        # normalized overlaps
        ox = dot(conj(u00), u10)
        oy = dot(conj(u00), u01)
        Ux[ix, iy] = ox / abs(ox)
        Uy[ix, iy] = oy / abs(oy)
    end

    # Berry curvature on each plaquette
    total = 0.0
    for ix in 1:Nk, iy in 1:Nk
        ixp = (ix % Nk) + 1
        iyp = (iy % Nk) + 1

        # F = arg( Ux(k) Uy(k+dx) / (Uy(k) Ux(k+dy)) )
        z = Ux[ix, iy] * Uy[ixp, iy] / (Uy[ix, iy] * Ux[ix, iyp])
        total += angle(z)
    end

    return total / (2π)
end

function main()
    Nk = 41
    m  = -1.0   # expected Chern ≈ +1 for this model
    Hk = (kx, ky) -> qwx_hamiltonian(kx, ky; m=m)

    C = chern_fhs(Hk; band=1, Nk=Nk)
    println("Nk = ", Nk, ", m = ", m, ", Chern ≈ ", C)
end

main()
