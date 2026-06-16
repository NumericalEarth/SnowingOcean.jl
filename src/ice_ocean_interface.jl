using Oceananigans
using Oceananigans: defaults
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, Center
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition
using Oceananigans.Utils: launch!
using Oceananigans.Fields: CenterField
using KernelAbstractions: @kernel, @index

using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus
import ClimaSeaIce.SeaIceThermodynamics: melting_temperature

#####
##### Depth-dependent liquidus (reuses the ClimaSeaIce salinity-dependent liquidus)
#####

"""
    DepthDependentLiquidus(liquidus=LinearLiquidus(); depth_slope=7.61e-4)

Augment a (salinity-dependent) `liquidus` with a linear depth dependence of the
freezing temperature:

```math
Tᶠ(S, z) = Tᶠ₀(S) + λ z ,
```

where `Tᶠ₀(S)` is the underlying liquidus (e.g. the ClimaSeaIce `LinearLiquidus`,
`Tᶠ₀(S) = T₀ - Γ S`), `z ≤ 0` is height, and `λ = depth_slope > 0` is the rate at
which the freezing point rises toward the surface. With the ClimaSeaIce
`LinearLiquidus` this gives the standard form

```math
Tᶠ(S, z) = T₀ - Γ S + λ z
```

The default `λ = 7.61e-4 °C m⁻¹` (Hewitt 2020, Annu. Rev. Fluid Mech., Table 1)
depresses the freezing point ≈0.76 °C at 1000 m depth.

This wrapper is exactly the depth dependence that could be upstreamed to ClimaSeaIce;
until then it lives here. It extends `melting_temperature` with a `(S, z)` method.
"""
struct DepthDependentLiquidus{L, FT}
    liquidus :: L
    depth_slope :: FT
end

function DepthDependentLiquidus(liquidus = LinearLiquidus(defaults.FloatType; slope=0.0573,
                                                          freshwater_melting_temperature=0.0832);
                                depth_slope = 7.61e-4)
    FT = defaults.FloatType
    return DepthDependentLiquidus(liquidus, convert(FT, depth_slope))
end

""" Freezing temperature at salinity `S` and height `z ≤ 0`. """
@inline melting_temperature(l::DepthDependentLiquidus, S, z) =
    melting_temperature(l.liquidus, S) + l.depth_slope * z

#####
##### Melt formulations
#####

"""
    AbstractMeltFormulation

Supertype for the ice–ocean melt parameterizations [`TwoEquation`](@ref) and
[`ThreeEquation`](@ref).
"""
abstract type AbstractMeltFormulation end

"""
    ThreeEquation()

The three-equation ice–ocean melt formulation (Holland & Jenkins 1999). The interface
temperature `Tᵦ` and salinity `Sᵦ` and the melt rate `ṁ` (m s⁻¹, > 0 melting) satisfy
three constraints — a heat balance, a salt balance, and the liquidus — where the ocean
turbulent exchange velocities are `γ_T = Γ_T u★` and `γ_S = Γ_S u★`:

```math
ρₒ cₒ γ_T (T - Tᵦ) = ρᵢ L ṁ          (heat balance, insulating ice)
ρₒ γ_S (S - Sᵦ)    = ρᵢ ṁ Sᵦ         (salt balance, zero ice salinity)
Tᵦ                 = Tᶠ(Sᵦ, z)        (liquidus)
```

With a linear liquidus `Tᶠ(S,z) = T₀ - Γ S + λ z` this closes as a **quadratic** in the
interface salinity `Sᵦ` (no iteration required):

```math
A Sᵦ² + B Sᵦ + C = 0,
A = -Γ κ,  B = -(κ (T - b) + γ_S),  C = γ_S S,
```

where `κ = cₒ γ_T / L` and `b = T₀ + λ z`. The physical (positive) root is
`Sᵦ = (-B - √(B² - 4AC)) / (2A)`; then `Tᵦ = Tᶠ(Sᵦ, z)` and
`ṁ = (ρₒ/ρᵢ) κ (T - Tᵦ)`.
"""
struct ThreeEquation <: AbstractMeltFormulation end

"""
    TwoEquation()

The two-equation ice–ocean melt formulation (McPhee et al. 2008). The salt balance is
dropped and the interface salinity is set to the far-field ocean salinity, `Sᵦ = S`, so
the interface sits at the freezing point of the ambient water, `Tᵦ = Tᶠ(S, z)`. The melt
rate then follows explicitly from the heat balance,

```math
ṁ = (ρₒ/ρᵢ) (cₒ γ_T / L) (T - Tᶠ(S, z)) ,    γ_T = Γ_T u★ .
```

This neglects the buffering of the interface by accumulated meltwater (`Sᵦ < S`) and so
generally overestimates melt relative to [`ThreeEquation`](@ref); the two are close for
shallow ice bases / weak thermal driving.
"""
struct TwoEquation <: AbstractMeltFormulation end

#####
##### The interface solve
#####

"""
    interface_temperature_and_salinity(formulation, liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)

Return the ice–ocean interface temperature `Tᵦ`, interface salinity `Sᵦ`, and melt rate
`ṁ` (m s⁻¹, positive for melting) for the given `formulation` ([`TwoEquation`](@ref) or
[`ThreeEquation`](@ref)), `liquidus`, adjacent ocean temperature `T` and salinity `S`,
height `z`, friction velocity `u★`, thermal/haline transfer coefficients `Γ_T, Γ_S`,
ocean heat capacity `c`, latent heat `L`, and densities `ρₒ, ρᵢ`. See [`ThreeEquation`](@ref)
and [`TwoEquation`](@ref) for the governing equations.
"""
@inline function interface_temperature_and_salinity(::ThreeEquation, liquidus, T, S, z, u★,
                                                     Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
    γ_T = Γ_T * u★
    γ_S = Γ_S * u★

    b = melting_temperature(liquidus, zero(S), z)                  # intercept T₀ + λ z (S = 0)
    a = melting_temperature(liquidus, oneunit(S), z) - b           # dTᵦ/dSᵦ = -Γ

    κ = c * γ_T / L
    A = κ * a
    B = - (κ * (T - b) + γ_S)
    C = γ_S * S
    Δ = max(B^2 - 4A * C, zero(B))
    Sᵦ = (- B - sqrt(Δ)) / (2A)                                    # physical (positive) root

    Tᵦ = b + a * Sᵦ
    ṁ = ρₒ / ρᵢ * κ * (T - Tᵦ)

    return Tᵦ, Sᵦ, ṁ
end

@inline function interface_temperature_and_salinity(::TwoEquation, liquidus, T, S, z, u★,
                                                     Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
    γ_T = Γ_T * u★
    Sᵦ = S
    Tᵦ = melting_temperature(liquidus, S, z)
    ṁ = ρₒ / ρᵢ * c * γ_T / L * (T - Tᵦ)
    return Tᵦ, Sᵦ, ṁ
end

#####
##### The interface object: holds the flux Fields, the drag, and the melt parameters
#####

"""
    IceOceanInterface(grid; kwargs...)

A utility that bundles an ice–ocean melt parameterization with the boundary-flux
`Field`s it fills. It holds the kinematic temperature flux `Jᵀ` and salt flux `Jˢ`
(`CenterField`s on `grid`) together with the drag coefficient and transfer coefficients,
so the melt fluxes are computed with the *same* friction velocity `u★ = √(Cᴰ |u|²)` used
by the `BulkDrag` momentum boundary condition (hence "consistent with the drag").

Keyword arguments (defaults follow Hewitt 2020 Table 1 / ISOMIP+; types follow
`Oceananigans.defaults.FloatType` via the `grid`):
- `formulation = ThreeEquation()` (or `TwoEquation()`)
- `drag_coefficient = 2.5e-3` (`Cᴰ`)
- `minimum_friction_velocity = 1e-3` (floor on `u★`, m s⁻¹)
- `heat_transfer_coefficient = 0.022` (`Γ_T`; `St_T = √Cᴰ Γ_T ≈ 1.1e-3`)
- `salt_transfer_coefficient = 6.2e-4` (`Γ_S`; ratio `Γ_T/Γ_S ≈ 35`)
- `liquidus = DepthDependentLiquidus()`
- `ocean_reference_density = 1028.0`, `ocean_heat_capacity = 3974.0`
- `ice_density = 917.0`, `latent_heat = 3.34e5`

Build the boundary conditions with [`ice_ocean_boundary_conditions`](@ref) and keep the
flux fields current with [`add_melt_flux_callback!`](@ref) / [`compute_melt_fluxes!`](@ref).
"""
struct IceOceanInterface{FM, FT, L, F}
    formulation :: FM
    drag_coefficient :: FT
    minimum_friction_velocity :: FT
    heat_transfer_coefficient :: FT
    salt_transfer_coefficient :: FT
    liquidus :: L
    ocean_reference_density :: FT
    ocean_heat_capacity :: FT
    ice_density :: FT
    latent_heat :: FT
    temperature_flux :: F
    salt_flux :: F
end

function IceOceanInterface(grid;
                           formulation = ThreeEquation(),
                           drag_coefficient = 2.5e-3,
                           minimum_friction_velocity = 1e-3,
                           heat_transfer_coefficient = 0.022,
                           salt_transfer_coefficient = 6.2e-4,
                           liquidus = DepthDependentLiquidus(),
                           ocean_reference_density = 1028.0,
                           ocean_heat_capacity = 3974.0,
                           ice_density = 917.0,
                           latent_heat = 3.34e5)

    FT = eltype(grid)
    Jᵀ = CenterField(grid)
    Jˢ = CenterField(grid)

    return IceOceanInterface(formulation,
                             convert(FT, drag_coefficient),
                             convert(FT, minimum_friction_velocity),
                             convert(FT, heat_transfer_coefficient),
                             convert(FT, salt_transfer_coefficient),
                             liquidus,
                             convert(FT, ocean_reference_density),
                             convert(FT, ocean_heat_capacity),
                             convert(FT, ice_density),
                             convert(FT, latent_heat),
                             Jᵀ, Jˢ)
end

#####
##### Computing the fluxes into the Fields (in a callback)
#####

@kernel function _compute_melt_fluxes!(Jᵀ, Jˢ, grid, T, S, u, v, p)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Tᵢ = T[i, j, k]
        Sᵢ = S[i, j, k]
        uᶜ = ℑxᶜᵃᵃ(i, j, k, grid, u)
        vᶜ = ℑyᵃᶜᵃ(i, j, k, grid, v)
        u★ = max(sqrt(p.drag_coefficient * (uᶜ^2 + vᶜ^2)), p.minimum_friction_velocity)
        z = znode(i, j, k, grid, Center(), Center(), Center())

        Tᵦ, Sᵦ, ṁ = interface_temperature_and_salinity(p.formulation, p.liquidus, Tᵢ, Sᵢ, z, u★,
                                                        p.heat_transfer_coefficient,
                                                        p.salt_transfer_coefficient,
                                                        p.ocean_heat_capacity, p.latent_heat,
                                                        p.ocean_reference_density, p.ice_density)

        γ_T = p.heat_transfer_coefficient * u★
        γ_S = p.salt_transfer_coefficient * u★

        # Kinematic fluxes applied at the (immersed top) interface. The boundary-condition
        # value increases the boundary-adjacent cell, so melting (T > Tᵦ) must give a
        # negative temperature flux to cool the ocean: Jᵀ = -γ_T (T - Tᵦ). Likewise
        # meltwater (Sᵦ < S) freshens the ocean: Jˢ = -γ_S (S - Sᵦ).
        Jᵀ[i, j, k] = - γ_T * (Tᵢ - Tᵦ)
        if p.formulation isa TwoEquation
            Jˢ[i, j, k] = - p.ice_density / p.ocean_reference_density * ṁ * Sᵢ
        else
            Jˢ[i, j, k] = - γ_S * (Sᵢ - Sᵦ)
        end
    end
end

"""
    compute_melt_fluxes!(model, interface::IceOceanInterface)

Fill `interface.temperature_flux` and `interface.salt_flux` from the current model state
(`T`, `S`, `u`, `v`). Designed to be called in a callback; see [`add_melt_flux_callback!`](@ref).
"""
function compute_melt_fluxes!(model, interface::IceOceanInterface)
    grid = model.grid
    arch = architecture(grid)

    T = model.tracers.T
    S = model.tracers.S
    u = model.velocities.u
    v = model.velocities.v

    p = (; interface.formulation,
           interface.drag_coefficient,
           interface.minimum_friction_velocity,
           interface.heat_transfer_coefficient,
           interface.salt_transfer_coefficient,
           interface.liquidus,
           interface.ocean_heat_capacity,
           interface.latent_heat,
           interface.ocean_reference_density,
           interface.ice_density)

    launch!(arch, grid, :xyz, _compute_melt_fluxes!,
            interface.temperature_flux, interface.salt_flux, grid, T, S, u, v, p)

    return nothing
end

"""
    add_melt_flux_callback!(simulation, interface; name=:melt_fluxes)

Add a callback that calls [`compute_melt_fluxes!`](@ref) every iteration, keeping the flux
`Field`s consistent with the evolving ocean state.
"""
function add_melt_flux_callback!(simulation, interface::IceOceanInterface; name=:melt_fluxes)
    simulation.callbacks[name] = Callback(sim -> compute_melt_fluxes!(sim.model, interface),
                                          IterationInterval(1))
    return simulation
end

#####
##### Boundary conditions that read the flux Fields
#####

# Discrete-form flux on an immersed boundary: index the precomputed flux Field directly.
# (Oceananigans `getbc` only supports 2D array conditions [i,j]; on a 3D immersed boundary
# we must use a discrete function that indexes [i,j,k].)
@inline ice_flux(i, j, k, grid, clock, fields, J) = @inbounds J[i, j, k]
@inline ice_flux(i, j, grid, clock, fields, J)    = @inbounds J[i, j]

"""
    ice_ocean_boundary_conditions(interface; drag=true)

Return a `NamedTuple` of `FieldBoundaryConditions` `(u, v, T, S)` that apply the ice–ocean
melt fluxes on the immersed boundary (ice base). The `T` and `S` conditions read
`interface.temperature_flux`/`salt_flux`; the `u`, `v` conditions apply `BulkDrag`
with the same `drag_coefficient`, so momentum and melt fluxes are mutually consistent. Pass
`drag=false` to omit the momentum drag and return only `(T, S)`.
"""
function ice_ocean_boundary_conditions(interface::IceOceanInterface; drag::Bool=true)
    Jᵀ = interface.temperature_flux
    Jˢ = interface.salt_flux

    T_flux = FluxBoundaryCondition(ice_flux; discrete_form=true, parameters=Jᵀ)
    S_flux = FluxBoundaryCondition(ice_flux; discrete_form=true, parameters=Jˢ)

    T_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = T_flux))
    S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = S_flux))

    if drag
        u_drag = BulkDrag(coefficient = interface.drag_coefficient)
        v_drag = BulkDrag(coefficient = interface.drag_coefficient)
        u_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = u_drag))
        v_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = v_drag))
        return (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)
    else
        return (T = T_bcs, S = S_bcs)
    end
end
