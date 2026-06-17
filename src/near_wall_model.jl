#####
##### Monin–Obukhov near-wall model for the ice-melting boundary condition
#####
##### After Vreugdenhil et al. (2022, J. Phys. Oceanogr. 52, 1903; Appendix B): a wall model
##### for large-eddy simulations that places the first grid point in the logarithmic layer
##### and uses Monin–Obukhov similarity to compute the friction velocity and the heat/salt
##### transfer coefficients *self-consistently with the local stratification*, coupled to the
##### three-equation diffusive melt closure. It generalizes [`ThreeEquation`](@ref): instead
##### of prescribing constant `Cᴰ, Γ_T, Γ_S`, those follow from the resolved first-cell speed,
##### temperature, and salinity and the buoyancy-driven Obukhov length. The constant-
##### coefficient three-equation model is recovered in the weak-stratification limit `L → ∞`.
#####

"""
    MoninObukhovNearWall(FT=Oceananigans.defaults.FloatType; kwargs...)

The Monin–Obukhov near-wall melt formulation of Vreugdenhil et al. (2022). Used as the
`formulation` of an [`IceOceanInterface`](@ref). The friction velocity `u★` and the thermal
and haline transfer coefficients are obtained by solving the law-of-the-wall profiles
(with the linear stratification correction `Φ = 1 + β ξ`, `ξ = z_g / L`, Obukhov length
`L = -u★³ / (k_m B)`, buoyancy flux `B = g u★ (α T★ - β S★)`) together with the
three-equation heat/salt/liquidus closure. The system is solved by fixed-point iteration
seeded from quadratic drag; if it fails to produce a finite result it falls back to the
constant-coefficient [`ThreeEquation`](@ref) solve.

Keyword arguments (defaults after Vreugdenhil et al. 2022):
- `momentum_von_karman = 0.41`, `scalar_von_karman = 0.48`
- `momentum_stability = 4.8`, `scalar_stability = 5.6` (linear MOST `Φ` slopes)
- `kinematic_viscosity = 1.8e-6`, `prandtl = 13.8`, `schmidt = 2432`
- `thermal_expansion = 3.28e-5`, `haline_contraction = 7.84e-4`
- `gravitational_acceleration = 9.81`
- `iterations = 10`
"""
struct MoninObukhovNearWall{FT} <: AbstractMeltFormulation
    momentum_von_karman :: FT
    scalar_von_karman :: FT
    momentum_stability :: FT
    scalar_stability :: FT
    kinematic_viscosity :: FT
    prandtl :: FT
    schmidt :: FT
    thermal_expansion :: FT
    haline_contraction :: FT
    gravitational_acceleration :: FT
    iterations :: Int
end

function MoninObukhovNearWall(FT::DataType = defaults.FloatType;
                              momentum_von_karman = 0.41,
                              scalar_von_karman = 0.48,
                              momentum_stability = 4.8,
                              scalar_stability = 5.6,
                              kinematic_viscosity = 1.8e-6,
                              prandtl = 13.8,
                              schmidt = 2432,
                              thermal_expansion = 3.28e-5,
                              haline_contraction = 7.84e-4,
                              gravitational_acceleration = 9.81,
                              iterations = 10)

    return MoninObukhovNearWall{FT}(convert(FT, momentum_von_karman),
                                    convert(FT, scalar_von_karman),
                                    convert(FT, momentum_stability),
                                    convert(FT, scalar_stability),
                                    convert(FT, kinematic_viscosity),
                                    convert(FT, prandtl),
                                    convert(FT, schmidt),
                                    convert(FT, thermal_expansion),
                                    convert(FT, haline_contraction),
                                    convert(FT, gravitational_acceleration),
                                    iterations)
end

@inline function solve_interface(nwm::MoninObukhovNearWall, p, T, S, speed, z, z_g)
    ν  = nwm.kinematic_viscosity
    kₘ = nwm.momentum_von_karman
    kₛ = nwm.scalar_von_karman
    βₘ = nwm.momentum_stability
    βₛ = nwm.scalar_stability
    g  = nwm.gravitational_acceleration
    α  = nwm.thermal_expansion
    βS = nwm.haline_contraction

    c  = p.ocean_heat_capacity
    L  = p.latent_heat
    ρₒ = p.ocean_reference_density
    ρᵢ = p.ice_density
    u★min = p.minimum_friction_velocity

    # Sublayer integration constants for the temperature and salinity profiles
    Cᵀ = 13 * nwm.prandtl^(2//3) - 15//2
    Cˢ = 13 * nwm.schmidt^(2//3)  - 15//2

    U  = max(speed, u★min)
    u★ = max(sqrt(p.drag_coefficient) * U, u★min)  # initial guess from quadratic drag
    ξ  = zero(u★)                                   # neutral (unstratified) to start

    Tᵦ = melting_temperature(p.liquidus, S, z)
    Sᵦ = S
    ṁ  = zero(u★)

    @inbounds for _ in 1:nwm.iterations
        ℓ = log(z_g * u★ / ν)
        fₘ = max((ℓ + βₘ * ξ) / kₘ + 5, one(u★))    # guard against a non-positive profile factor
        u★ = max(U / fₘ, u★min)

        fᵀ = (ℓ + βₛ * ξ) / kₛ + Cᵀ
        fˢ = (ℓ + βₛ * ξ) / kₛ + Cˢ

        # The three-equation closure with the MOST transfer coefficients Γ_T = 1/fᵀ, Γ_S = 1/fˢ
        Tᵦ, Sᵦ, ṁ = interface_temperature_and_salinity(ThreeEquation(), p.liquidus, T, S, z, u★,
                                                        inv(fᵀ), inv(fˢ), c, L, ρₒ, ρᵢ)

        T★ = (T - Tᵦ) / fᵀ                          # friction temperature
        S★ = (S - Sᵦ) / fˢ                          # friction salinity
        ξ  = - z_g * kₘ * g * (α * T★ - βS * S★) / u★^2
    end

    # Fall back to the constant-coefficient three-equation solve if the iteration diverged
    if !isfinite(ṁ) || !isfinite(Sᵦ)
        return solve_interface(ThreeEquation(), p, T, S, speed, z, z_g)
    end

    return Tᵦ, Sᵦ, ṁ, u★
end
