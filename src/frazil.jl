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

A bulk model for the generation and melting of frazil ice in supercooled water. The ice
volume fraction `ϕ` and temperature `T` relax toward thermodynamic equilibrium (the local
freezing point `T⋆ = Tᶠ(S, z)`) at a rate set by `1/timescale`:

```math
\\frac{\\mathrm{D} T}{\\mathrm{D} t} = \\frac{1}{τ}\\,(T⋆ - T), \\qquad
\\frac{\\mathrm{D} ϕ}{\\mathrm{D} t} = \\frac{1}{τ\\,𝒯}\\,(T⋆ - T),
```

where `τ = timescale` and `𝒯 = L / c` is a temperature scale built from the latent heat of
fusion `L` and heat capacity `c`. Supercooled water (`T < T⋆`) grows frazil (`Dϕ/Dt > 0`),
releasing latent heat that warms the water back toward freezing; warm water (`T > T⋆`) melts
existing frazil. Because `c \\, (T⋆ - T)/τ = L \\, (T⋆ - T)/(τ 𝒯)`, i.e. `c\\,F_T = L\\,F_ϕ`,
the source terms exactly conserve the combined sensible-plus-latent energy `c\\,T - L\\,ϕ`.

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
`Fᵀ = (T⋆ - T)/τ` and `Fᵠ = Fᵀ/𝒯` with `𝒯 = L/c` and `T⋆ = Tᶠ(S, z)`.
"""
@inline function frazil_tendencies(frazil::FrazilModel, T, S, z)
    T★ = melting_temperature(frazil.liquidus, S, z)
    𝒯 = frazil.latent_heat / frazil.heat_capacity
    Fᵀ = (T★ - T) / frazil.timescale
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
