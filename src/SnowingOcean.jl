module SnowingOcean

export
    monochromatic_stokes_drift,
    rising_tracer_forcing,
    linear_freezing_temperature,
    LinearLiquidus,
    DepthDependentLiquidus,
    melting_temperature,
    TwoEquation,
    ThreeEquation,
    interface_temperature_and_salinity,
    IceOceanInterface,
    compute_melt_fluxes!,
    add_melt_flux_callback!,
    ice_ocean_boundary_conditions

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition

"""
    monochromatic_stokes_drift(; amplitude, wavelength, gravitational_acceleration=9.81)

Return a `UniformStokesDrift` for a monochromatic, deep-water surface gravity
wave with the given `amplitude` and `wavelength` (both in meters).

The deep-water dispersion relation ``σ = √(g k)`` is used, where ``k = 2π /
wavelength`` is the wavenumber and ``g`` is `gravitational_acceleration`. The
surface Stokes drift velocity scale is ``Uˢ = a² σ k``.
"""
function monochromatic_stokes_drift(; amplitude, wavelength, gravitational_acceleration=9.81)
    a = amplitude
    k = 2π / wavelength               # wavenumber (m⁻¹)
    σ = sqrt(gravitational_acceleration * k) # angular frequency (s⁻¹)
    Uˢ = a^2 * σ * k                  # surface Stokes drift scale (m s⁻¹)

    @inline ∂z_uˢ(z, t, p) = 1 / (2 * p.k) * p.Uˢ * exp(2 * p.k * z)

    return UniformStokesDrift(∂z_uˢ=∂z_uˢ, parameters=(; k, Uˢ))
end

"""
    rising_tracer_forcing(grid, w)

Return an `AdvectiveForcing` that advects a tracer upward at constant velocity
`w` (m s⁻¹, positive upward). The vertical velocity field is given
`ImpenetrableBoundaryCondition`s at the top and bottom so that total tracer is
conserved.
"""
function rising_tracer_forcing(grid, w)
    w_location = (Face(), Center(), Center())
    w_bcs = FieldBoundaryConditions(grid, w_location,
                                    top = ImpenetrableBoundaryCondition(),
                                    bottom = ImpenetrableBoundaryCondition())

    wc = ZFaceField(grid, boundary_conditions=w_bcs)
    set!(wc, w)
    fill_halo_regions!(wc)

    return AdvectiveForcing(w=wc)
end

"""
    linear_freezing_temperature(S, z; slope=0.0545, depth_coefficient=7.9e-4)

Linear approximation of the freezing temperature (°C) of seawater with absolute
salinity `S` (g kg⁻¹) at depth `z` (m, negative below the surface):

```math
Tᶠ = - slope * S + depth_coefficient * z
```

The default coefficients approximate the TEOS-10 freezing temperature
`GibbsSeaWater.gsw_ct_freezing` over the upper kilometer of the ocean.
"""
@inline linear_freezing_temperature(S, z; slope=0.0545, depth_coefficient=7.9e-4) =
    - slope * S + depth_coefficient * z

include("ice_ocean_interface.jl")

end # module SnowingOcean
