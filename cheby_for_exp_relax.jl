module Cheby

using SpecialFunctions: besseli, besselj
using LinearAlgebra

export ChebyWrk, cheby!


"""
Workspace for the Chebychev propagation routine.

```julia
ChebyWrk(v, Δ, E_min, E_max, dt; limit=1e-12)
```

initializes the workspace for the propagation of a state similar to `v` under a
Operator with eigenvalues between `E_min` and `E_min + Δ`,
and a time step `dt`. Chebychev coefficients smaller than the given `limit` are
discarded.
"""
mutable struct ChebyWrk{ST,CFS,FT<:AbstractFloat}
    v0::ST
    v1::ST
    v2::ST
    coeffs::CFS
    n_coeffs::Int64
    Δ::FT
    E_min::FT
    dt::FT
    limit::FT
    function ChebyWrk(
        v::ST,
        E_min::FT,
        E_max::FT,
        dt::FT;
        limit::FT=1e-12,
    ) where {ST,FT}
        Δ = E_max - E_min
        v0::ST = similar(v)
        v1::ST = similar(v)
        v2::ST = similar(v)
        coeffs = cheby_coeffs(Δ, dt; limit=limit)
        n_coeffs = length(coeffs)
        new{ST,typeof(coeffs),FT}(
            v0,
            v1,
            v2,
            coeffs,
            n_coeffs,
            Δ,
            E_min,
            dt,
            limit,
        )
    end
end


"""Calculate Chebychev coefficients.

```julia
a::Vector{Float64} = cheby_coeffs(Δ, dt; limit=1e-12)
```

return an array of coefficiencts larger than `limit`.

# Arguments

* `Δ`: the spectral radius of the underlying operator
* `dt`: the time step

See also [`cheby_coeffs!`](@ref) for an in-place version.
"""
function cheby_coeffs(Δ, dt; limit=1e-12)
    α = abs(0.5 * Δ * dt)
    coeffs = Float64[]
    a = besseli(0, α)
    append!(coeffs, a)
    ϵ = abs(a)
    i = 1
    while ϵ > limit
        a = 2 * besseli(i, α)
        append!(coeffs, a)
        ϵ = abs(a)
        i += 1
    end
    return coeffs
end


"""Evaluate `Ψ = exp(- H dt) Ψ` in-place.

```julia
cheby!(Ψ, H, dt, wrk; E_min=nothing, check_normalization=false)
```

# Arguments

* `Ψ`: on input, initial vector. Will be overwritten with result.
* `H`: Hermitian operator
* `dt`: time step
* `wrk`: internal workspace
* `E_min`: minimum eigenvalue of H, to be used instead of the `E_min` from the
   initialization of `wrk`. The same `wrk` may be used for different values
   `E_min`, as long as the spectra radius `Δ` and the time step `dt` are the
   same as those used for the initialization of `wrk`.
* `check_normalizataion`: perform checks that the H does not exceed the
  spectral radius for which the the workspace was initialized.

The routine will not allocate any internal storage. This implementation
requires `copyto!` `lmul!`, and `axpy!` to be implemented for `Ψ`, and the
three-argument `mul!` for `Ψ` and `H`.
"""
function cheby!(Ψ, H, dt, wrk; kwargs...)

    E_min = get(kwargs, :E_min, wrk.E_min)
    check_normalization = get(kwargs, :check_normalization, false)

    Δ = wrk.Δ
    β::Float64 = (Δ / 2) + E_min  # "normfactor"
    @assert abs(dt) ≈ abs(wrk.dt) "wrk was initialized for dt=$(wrk.dt), not dt=$dt"
    if dt > 0
        c = -2 / Δ
    else
        c = 2 / Δ
    end
    a = wrk.coeffs
    ϵ = wrk.limit
    @assert length(a) > 1 "Need at least 2 Chebychev coefficients"
    v0 = wrk.v0
    v1 = wrk.v1
    v2 = wrk.v2

    # v0 ↔ Ψ; Ψ = a[1] * v0
    copyto!(v0, Ψ)
    lmul!(a[1], Ψ)

    # v1 = -i * H_norm * v0 = c * (H * v0 - β * v0)
    mul!(v1, H, v0)
    axpy!(-β, v0, v1)
    lmul!(c, v1)

    # Ψ += a[2] * v1
    axpy!(a[2], v1, Ψ)

    c *= 2

    for i = 3:wrk.n_coeffs

        # v2 = -2i * H_norm * v1 + v0 = c * (H * v1 - β * v1) + v0
        mul!(v2, H, v1)
        axpy!(-β, v1, v2)
        lmul!(c, v2)
        if check_normalization
            map_norm = abs(dot(v1, v2)) / (2 * norm(v1)^2)
            @assert(
                map_norm <= (1.0 + ϵ),
                "Incorrect normalization (E_min=$(E_min), Δ=$(Δ))"
            )
        end
        # v2 += v0
        axpy!(true, v0, v2)

        # Ψ += a[i] * v2
        axpy!(a[i], v2, Ψ)

        v0, v1, v2 = v1, v2, v0  # switch w/o copying

    end

    lmul!(exp(-β * dt), Ψ)

end


end

# =============================================================================

using Random

"""Construct a random Hermitian matrix of size N×N with spectral radius ρ.

```julia
random_hermitian_matrix(N, ρ)
```
"""
function random_hermitian_matrix(N, ρ; rng=Random.GLOBAL_RNG, exact_spectral_radius=false)
    Δ = √(12 / N)
    X = Δ * (rand(rng, N, N) .- 0.5)
    H = ρ * (X + X') / (2 * √2)
    if exact_spectral_radius
        λ = eigvals(H)
        λ₀ = λ[1]
        Δ = λ[end] - λ₀
        H_norm = (H - λ₀ * I) / Δ
        return 2 * ρ * H_norm - ρ * I
    else
        return H
    end
end

using LinearAlgebra

using .Cheby

N = 10  # dimension
ρ = 1.0 # spectral radius

Q = random_hermitian_matrix(N, ρ; exact_spectral_radius=true) + 1im * zeros(N, N)
E_min = -1.01*ρ
E_max = 1.01ρ
@assert E_min < eigvals(Q)[begin]
@assert E_max > eigvals(Q)[end]
dt = 2.0  # time step

v = rand(N) + 1im * zeros(N)
v /= norm(v)  # just to make testing easier

wrk = ChebyWrk(v, E_min, E_max, dt; limit=1e-14)

groundstate = eigen(Q).vectors[:,1]
@assert (1 - norm(groundstate)) < 1e-12

expected = exp(-1 * Q * dt) * v

@show norm(expected / norm(expected) - groundstate) # → 0 for dt → ∞

v_out = copy(v)
cheby!(v_out, Q, dt, wrk)

using Test
@test norm(v_out - expected) < 1e-12
