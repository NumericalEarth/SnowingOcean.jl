using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, Center
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions, ImmersedBoundaryCondition
using Oceananigans.Utils: launch!
using Oceananigans.Fields: CenterField
using KernelAbstractions: @kernel, @index

#####
##### Linear liquidus (freezing temperature), depth-dependent
#####

"""
    LinearLiquidus(FT=Float64; reference_temperature, haline_slope, depth_slope)

A linear approximation of the seawater freezing temperature,

```math
Tᶠ(S, z) = T₀ - Γ S + λ z
```

where `S` is salinity (g kg⁻¹), `z ≤ 0` is height (m), `T₀ = reference_temperature`,
`Γ = haline_slope`, and `λ = depth_slope`. The defaults follow Table 1 of Hewitt
(2020, "Subglacial Plumes", Annu. Rev. Fluid Mech.): `T₀ = 0.0832 °C`,
`Γ = 0.0573 °C (g/kg)⁻¹`, `λ = 7.61e-4 °C m⁻¹` (freezing point depressed ≈0.76 °C
at 1000 m depth). Compatible with the ISOMIP+/Jenkins et al. (2010) coefficients.
"""
struct LinearLiquidus{FT}
    reference_temperature :: FT
    haline_slope :: FT
    depth_slope :: FT
end

function LinearLiquidus(FT::DataType = Float64;
                        reference_temperature = 0.0832,
                        haline_slope = 0.0573,
                        depth_slope = 7.61e-4)

    return LinearLiquidus{FT}(convert(FT, reference_temperature),
                              convert(FT, haline_slope),
                              convert(FT, depth_slope))
end

""" Freezing temperature at salinity `S` and height `z` (≤ 0). """
@inline melting_temperature(l::LinearLiquidus, S, z) =
    l.reference_temperature - l.haline_slope * S + l.depth_slope * z

#####
##### The three-equation (and two-equation) interface solve
#####

"""
    interface_temperature_and_salinity(formulation, liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)

Return the ice–ocean interface temperature `Tᵦ`, interface salinity `Sᵦ`, and melt
rate `ṁ` (m s⁻¹, positive for melting) given the adjacent ocean temperature `T`,
salinity `S`, height `z`, friction velocity `u★`, thermal and haline transfer
coefficients `Γ_T, Γ_S`, ocean heat capacity `c`, latent heat `L`, and densities
`ρₒ, ρᵢ`.

The interface is assumed to be at the local freezing point (`Tᵦ = Tᶠ(Sᵦ, z)`). With
a linear liquidus the `:three_equation` heat + salt balance reduces to a closed-form
quadratic in `Sᵦ` (no iteration). The `:two_equation` formulation instead sets
`Sᵦ = S` (interface at the freezing point of the far-field salinity).
"""
@inline function interface_temperature_and_salinity(formulation, liquidus, T, S, z, u★,
                                                     Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
    γ_T = Γ_T * u★
    γ_S = Γ_S * u★

    a = - liquidus.haline_slope                                         # dTᵦ/dSᵦ
    b = liquidus.reference_temperature + liquidus.depth_slope * z       # intercept incl. depth

    if formulation === Val(:two_equation)
        Sᵦ = S
    else # :three_equation -- closed-form quadratic A Sᵦ² + B Sᵦ + C = 0
        κ = c * γ_T / L                       # melt-rate factor (kinematic)
        A = κ * a
        B = - (κ * (T - b) + γ_S)
        C = γ_S * S
        Δ = max(B^2 - 4A * C, zero(B))
        Sᵦ = (- B - sqrt(Δ)) / (2A)           # physical (positive) root, see notes
    end

    Tᵦ = b + a * Sᵦ
    ṁ = ρₒ / ρᵢ * c * γ_T / L * (T - Tᵦ)      # melt rate (m s⁻¹), > 0 melting

    return Tᵦ, Sᵦ, ṁ
end

#####
##### The interface object: holds the flux Fields and the drag/transfer parameters
#####

"""
    ThreeEquationInterface(grid; kwargs...)

A utility that bundles the ice–ocean melt parameterization with the boundary-flux
`Field`s it fills. It holds the kinematic temperature flux `Jᵀ` and salt flux `Jˢ`
(`CenterField`s on `grid`) together with the drag coefficient and transfer
coefficients, so that the melt fluxes are computed with the *same* friction velocity
`u★ = √(Cᴰ |u|²)` used by the [`BulkDrag`](@ref) momentum boundary condition.

Keyword arguments (defaults follow Hewitt 2020 Table 1 / ISOMIP+):
- `drag_coefficient = 2.5e-3` (`Cᴰ`)
- `minimum_friction_velocity = 1e-3` (floor on `u★`, m s⁻¹)
- `heat_transfer_coefficient = 0.022` (`Γ_T`, so `St_T = √Cᴰ Γ_T ≈ 1.1e-3`)
- `salt_transfer_coefficient = 6.2e-4` (`Γ_S`, ratio `Γ_T/Γ_S ≈ 35`)
- `liquidus = LinearLiquidus(FT)`
- `ocean_reference_density = 1028.0`, `ocean_heat_capacity = 3974.0`
- `ice_density = 917.0`, `latent_heat = 3.34e5`
- `formulation = :three_equation` (or `:two_equation`)

Build the boundary conditions with [`ice_ocean_boundary_conditions`](@ref) and keep
the fluxes current with [`compute_melt_fluxes!`](@ref) (e.g. via
[`add_melt_flux_callback!`](@ref)).
"""
struct ThreeEquationInterface{FT, L, F, FM}
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
    formulation :: FM
end

function ThreeEquationInterface(grid;
                                drag_coefficient = 2.5e-3,
                                minimum_friction_velocity = 1e-3,
                                heat_transfer_coefficient = 0.022,
                                salt_transfer_coefficient = 6.2e-4,
                                liquidus = LinearLiquidus(eltype(grid)),
                                ocean_reference_density = 1028.0,
                                ocean_heat_capacity = 3974.0,
                                ice_density = 917.0,
                                latent_heat = 3.34e5,
                                formulation = :three_equation)

    FT = eltype(grid)
    Jᵀ = CenterField(grid)
    Jˢ = CenterField(grid)

    formulation ∈ (:two_equation, :three_equation) ||
        throw(ArgumentError("formulation must be :two_equation or :three_equation"))

    return ThreeEquationInterface(convert(FT, drag_coefficient),
                                  convert(FT, minimum_friction_velocity),
                                  convert(FT, heat_transfer_coefficient),
                                  convert(FT, salt_transfer_coefficient),
                                  liquidus,
                                  convert(FT, ocean_reference_density),
                                  convert(FT, ocean_heat_capacity),
                                  convert(FT, ice_density),
                                  convert(FT, latent_heat),
                                  Jᵀ, Jˢ,
                                  Val(formulation))
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

        # Upward (ocean → ice) kinematic fluxes. With the linear three-equation closure
        # the salt flux γ_S (S - Sᵦ) equals the meltwater dilution flux (ρᵢ/ρₒ) ṁ Sᵦ.
        Jᵀ[i, j, k] = γ_T * (Tᵢ - Tᵦ)
        if p.formulation === Val(:two_equation)
            Jˢ[i, j, k] = p.ice_density / p.ocean_reference_density * ṁ * Sᵢ
        else
            Jˢ[i, j, k] = γ_S * (Sᵢ - Sᵦ)
        end
    end
end

"""
    compute_melt_fluxes!(model, interface::ThreeEquationInterface)

Fill `interface.temperature_flux` and `interface.salt_flux` from the current model
state (`T`, `S`, `u`, `v`). Designed to be called in a callback; see
[`add_melt_flux_callback!`](@ref).
"""
function compute_melt_fluxes!(model, interface::ThreeEquationInterface)
    grid = model.grid
    arch = architecture(grid)

    T = model.tracers.T
    S = model.tracers.S
    u = model.velocities.u
    v = model.velocities.v

    p = (; interface.drag_coefficient,
           interface.minimum_friction_velocity,
           interface.heat_transfer_coefficient,
           interface.salt_transfer_coefficient,
           interface.liquidus,
           interface.ocean_heat_capacity,
           interface.latent_heat,
           interface.ocean_reference_density,
           interface.ice_density,
           interface.formulation)

    launch!(arch, grid, :xyz, _compute_melt_fluxes!,
            interface.temperature_flux, interface.salt_flux, grid, T, S, u, v, p)

    return nothing
end

"""
    add_melt_flux_callback!(simulation, interface; name=:melt_fluxes)

Add a callback that calls [`compute_melt_fluxes!`](@ref) every iteration, keeping the
flux `Field`s consistent with the evolving ocean state.
"""
function add_melt_flux_callback!(simulation, interface::ThreeEquationInterface; name=:melt_fluxes)
    simulation.callbacks[name] = Callback(sim -> compute_melt_fluxes!(sim.model, interface),
                                          IterationInterval(1))
    return simulation
end

#####
##### Boundary conditions that read the flux Fields
#####

# Discrete-form flux on an immersed boundary: index the precomputed flux Field directly.
# (Oceananigans `getbc` only supports 2D array conditions [i,j]; on a 3D immersed
# boundary we must use a discrete function that indexes [i,j,k].)
@inline ice_flux(i, j, k, grid, clock, fields, J) = @inbounds J[i, j, k]
@inline ice_flux(i, j, grid, clock, fields, J)    = @inbounds J[i, j]

"""
    ice_ocean_boundary_conditions(interface; drag=true)

Return a `NamedTuple` of `FieldBoundaryConditions` `(u, v, T, S)` that apply the
ice–ocean melt fluxes on the immersed boundary (ice base). The `T` and `S` conditions
read `interface.temperature_flux`/`salt_flux`; the `u`, `v` conditions apply
[`BulkDrag`](@ref) with the same `drag_coefficient`, so the momentum and melt fluxes
are mutually consistent. Pass `drag=false` to omit the momentum drag.
"""
function ice_ocean_boundary_conditions(interface::ThreeEquationInterface; drag::Bool=true)
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
