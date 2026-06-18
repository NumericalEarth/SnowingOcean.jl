#####
##### Frazil ice formation in supercooled turbulent flow
#####
#####
##### Following the model in "A model for frazil ice generation in supercooled
##### turbulent flow" (Snowing_ocean notes): the ice volume fraction ϕ grows when
##### the water is supercooled (T below the local freezing point T⋆), releasing
##### latent heat that drives T back toward T⋆.
#####

using Oceananigans: Forcing
using Oceananigans.Grids: znode, Center

"""
    FrazilModel(timescale; liquidus=DepthDependentLiquidus(), latent_heat=3.34e5, heat_capacity=3991.0)

A bulk model for the generation of frazil ice in supercooled water. Where the water is below
its local freezing point `T⋆ = Tᶠ(S, z)`, frazil grows and releases latent heat that relaxes
the temperature `T` back toward freezing while the ice volume fraction `ϕ` increases:

```math
\\frac{\\mathrm{D} T}{\\mathrm{D} t} = \\frac{\\max(T⋆ - T,\\, 0)}{τ}, \\qquad
\\frac{\\mathrm{D} ϕ}{\\mathrm{D} t} = \\frac{1}{𝒯}\\,\\frac{\\mathrm{D} T}{\\mathrm{D} t},
```

where `τ = timescale` and `𝒯 = L / c` is a temperature scale built from the latent heat of
fusion `L` and heat capacity `c`. The source **only heats** the ocean (frazil formation): warm
water (`T > T⋆`) produces no source, so the model neither cools the ocean spuriously nor drives
`ϕ` negative.

Energy is conserved because the two sources are locked together by `c\\,F_T = L\\,F_ϕ`: the
latent heat released equals the temperature rise, so frazil growth merely converts sensible heat
into latent heat. The combined sensible-plus-latent energy `e = c\\,T - L\\,ϕ` is therefore
unchanged by the source (`\\dot e = c F_T - L F_ϕ = 0`) — note this follows from the *coupling*
of the sources, not from `F_T` depending on `ϕ`.

Melting of frazil advected into warm water (`T > T⋆`) — which would *cool* the ocean and must be
rate-limited by the available `ϕ` — is deliberately not represented by this one-sided source.

The `timescale` slot is deliberately a free parameter. Here it is a constant; more complete
models tie the frazil-growth rate to the suspended-crystal population and geometry — the
crystal number density and mean size (Omstedt & Svensson 1984; Jenkins & Bombosch 1995;
Skyllingstad & Denbo 2001), with secondary (collisional) nucleation needed to sustain the
population and a salinity/diffusion correction to the effective supercooling (Rees Jones &
Wells 2018), and depth-dependent supercooling controlling where frazil concentrates
(Holland & Feltham 2005). Those can be supplied later as alternative `timescale` types.

Build the tracer/temperature forcings with [`frazil_forcing`](@ref).
"""
struct FrazilModel{TS, L, FT}
    timescale :: TS
    liquidus :: L
    latent_heat :: FT
    heat_capacity :: FT
end

function FrazilModel(timescale;
                     liquidus = DepthDependentLiquidus(),
                     latent_heat = 3.34e5,
                     heat_capacity = 3991.0)

    FT = defaults.FloatType
    return FrazilModel(timescale, liquidus, convert(FT, latent_heat), convert(FT, heat_capacity))
end

"""
    frazil_tendencies(frazil::FrazilModel, T, S, z)

Return the temperature and frazil-concentration source terms `(Fᵀ, Fᵠ)` at ocean
temperature `T`, salinity `S`, and height `z`:
`Fᵀ = max(T⋆ - T, 0)/τ` (heating from frazil growth, active only where supercooled) and
`Fᵠ = Fᵀ/𝒯` with `𝒯 = L/c` and `T⋆ = Tᶠ(S, z)`.
"""
@inline function frazil_tendencies(frazil::FrazilModel, T, S, z)
    T★ = melting_temperature(frazil.liquidus, S, z)
    𝒯 = frazil.latent_heat / frazil.heat_capacity
    supercooling = max(T★ - T, zero(T))      # only act where the water is below freezing
    Fᵀ = supercooling / frazil.timescale     # latent heating from frazil growth (≥ 0)
    Fᵠ = Fᵀ / 𝒯
    return Fᵀ, Fᵠ
end

# Discrete-form forcing wrappers (fixed arity, so they work on `Flat` grids and on the GPU).
@inline function _frazil_temperature_forcing(i, j, k, grid, clock, fields, frazil)
    @inbounds T = fields.T[i, j, k]
    @inbounds S = fields.S[i, j, k]
    z = znode(i, j, k, grid, Center(), Center(), Center())
    Fᵀ, _ = frazil_tendencies(frazil, T, S, z)
    return Fᵀ
end

@inline function _frazil_concentration_forcing(i, j, k, grid, clock, fields, frazil)
    @inbounds T = fields.T[i, j, k]
    @inbounds S = fields.S[i, j, k]
    z = znode(i, j, k, grid, Center(), Center(), Center())
    _, Fᵠ = frazil_tendencies(frazil, T, S, z)
    return Fᵠ
end

# Brine rejection: forming salt-free ice concentrates salt in the remaining liquid, so the
# liquid salinity rises at rate S × Dϕ/Dt (and falls when frazil melts). This is the
# dominant driver of buoyancy-forced convection during frazil formation.
@inline function _frazil_salt_forcing(i, j, k, grid, clock, fields, frazil)
    @inbounds T = fields.T[i, j, k]
    @inbounds S = fields.S[i, j, k]
    z = znode(i, j, k, grid, Center(), Center(), Center())
    _, Fᵠ = frazil_tendencies(frazil, T, S, z)
    return S * Fᵠ
end

"""
    frazil_forcing(frazil::FrazilModel; frazil_tracer=:ϕ, salt_rejection=true)

Return a `NamedTuple` of Oceananigans `Forcing`s implementing the [`FrazilModel`](@ref)
source terms, ready to pass as the `forcing` keyword to a model carrying `T`, `S`, and the
frazil tracer (named by `frazil_tracer`):

- a temperature source `Fᵀ = (T⋆ - T)/τ`,
- a frazil-concentration source `Fᵠ = Fᵀ/𝒯`, and
- (when `salt_rejection=true`) a salinity source `Fˢ = S\\,Fᵠ` representing brine rejection:
  growing salt-free ice concentrates salt in the remaining liquid, driving the haline
  convection that mixes frazil through the boundary layer.

The temperature/frazil sources conserve the combined sensible-plus-latent energy
`c\\,T - L\\,ϕ` regardless of `salt_rejection`.
"""
function frazil_forcing(frazil::FrazilModel; frazil_tracer::Symbol=:ϕ, salt_rejection::Bool=true)
    T_forcing = Forcing(_frazil_temperature_forcing; discrete_form=true, parameters=frazil)
    ϕ_forcing = Forcing(_frazil_concentration_forcing; discrete_form=true, parameters=frazil)
    forcing = merge((; T=T_forcing), NamedTuple{(frazil_tracer,)}((ϕ_forcing,)))
    if salt_rejection
        S_forcing = Forcing(_frazil_salt_forcing; discrete_form=true, parameters=frazil)
        forcing = merge(forcing, (; S=S_forcing))
    end
    return forcing
end
