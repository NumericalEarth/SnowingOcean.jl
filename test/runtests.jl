using SnowingOcean
using Oceananigans
using Test

@testset "SnowingOcean" begin

    @testset "liquidus (depth-dependent, reusing ClimaSeaIce)" begin
        # The ClimaSeaIce LinearLiquidus is salinity-only ...
        base = LinearLiquidus(slope=0.0573, freshwater_melting_temperature=0.0832)
        @test melting_temperature(base, 0) ≈ 0.0832
        @test melting_temperature(base, 35) < melting_temperature(base, 30)

        # ... and DepthDependentLiquidus adds the depth term.
        liquidus = DepthDependentLiquidus(base; depth_slope=7.61e-4)
        @test melting_temperature(liquidus, 0, 0) ≈ 0.0832
        @test melting_temperature(liquidus, 35, 0) < melting_temperature(liquidus, 30, 0)
        @test melting_temperature(liquidus, 34, -1000) < melting_temperature(liquidus, 34, 0)
        ΔT = melting_temperature(liquidus, 34, 0) - melting_temperature(liquidus, 34, -1000)
        @test ΔT ≈ 0.761 rtol=1e-3
    end

    @testset "interface solve" begin
        liquidus = DepthDependentLiquidus()
        Γ_T, Γ_S = 0.022, 6.2e-4
        c, L, ρₒ, ρᵢ = 3974.0, 3.34e5, 1028.0, 917.0
        u★ = 0.01
        S, z = 34.0, -200.0

        # Three-equation, warm ocean ⇒ melting: 0 < Sᵦ < S, interface at freezing, ṁ > 0
        T = 1.0
        Tᵦ, Sᵦ, ṁ = interface_temperature_and_salinity(ThreeEquation(), liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test 0 < Sᵦ < S
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, z)
        @test ṁ > 0
        @test Tᵦ < T

        # At the freezing point ⇒ no melt; supercooled ⇒ freezing
        Tf = melting_temperature(liquidus, S, z)
        _, _, ṁ0 = interface_temperature_and_salinity(ThreeEquation(), liquidus, Tf, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test abs(ṁ0) < 1e-10
        _, _, ṁf = interface_temperature_and_salinity(ThreeEquation(), liquidus, Tf - 0.1, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test ṁf < 0

        # Stronger thermal driving ⇒ more melt
        _, _, ṁ2 = interface_temperature_and_salinity(ThreeEquation(), liquidus, 2.0, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test ṁ2 > ṁ

        # Two-equation: interface salinity is the far-field salinity
        Tᵦ2, Sᵦ2, ṁ2eq = interface_temperature_and_salinity(TwoEquation(), liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test Sᵦ2 == S
        @test Tᵦ2 ≈ melting_temperature(liquidus, S, z)
        @test ṁ2eq > 0
        # Two-equation neglects interface freshening ⇒ overestimates melt
        @test ṁ2eq > ṁ
    end

    @testset "IceOceanInterface construction" begin
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        interface = IceOceanInterface(grid)
        @test interface.formulation isa ThreeEquation
        @test interface.temperature_flux isa Field
        @test interface.salt_flux isa Field
        @test all(interior(interface.temperature_flux) .== 0)
        @test all(interior(interface.salt_flux) .== 0)
        @test IceOceanInterface(grid; formulation=TwoEquation()).formulation isa TwoEquation
    end

    @testset "compute_melt_fluxes! fills the flux fields" begin
        grid = RectilinearGrid(size=(8, 1, 8), halo=(3, 1, 3),
                               topology=(Periodic, Periodic, Bounded), extent=(10, 10, 80))
        interface = IceOceanInterface(grid; minimum_friction_velocity=0.01)
        model = NonhydrostaticModel(grid; tracers=(:T, :S))
        set!(model, T=1.0, S=34.0)  # warm, above freezing
        compute_melt_fluxes!(model, interface)
        Jᵀ = interior(interface.temperature_flux)
        Jˢ = interior(interface.salt_flux)
        @test all(isfinite, Jᵀ) && all(isfinite, Jˢ)
        @test all(Jᵀ .!= 0) && all(Jˢ .!= 0)
    end

    @testset "immersed melt boundary condition cools and freshens the ocean" begin
        Lz = 8
        grid = RectilinearGrid(size=(8, 8), halo=(3, 3),
                               topology=(Periodic, Flat, Bounded),
                               x=(0, 8), z=(-Lz, 0))

        # Flat ice slab occupying the upper half of the domain
        ice(x, z) = z > -Lz/2
        ibg = ImmersedBoundaryGrid(grid, GridFittedBoundary(ice))

        interface = IceOceanInterface(ibg; minimum_friction_velocity=0.05)
        bcs = ice_ocean_boundary_conditions(interface)

        model = NonhydrostaticModel(ibg; tracers=(:T, :S),
                                    boundary_conditions=(; T=bcs.T, S=bcs.S, u=bcs.u, v=bcs.v))

        T₀, S₀ = 1.0, 34.0
        set!(model, T=T₀, S=S₀)

        simulation = Simulation(model, Δt=10.0, stop_iteration=5)
        add_melt_flux_callback!(simulation, interface)
        run!(simulation)

        T = interior(model.tracers.T)
        S = interior(model.tracers.S)
        # Melting of warm water should cool and freshen the ocean adjacent to the ice ...
        @test minimum(T) < T₀
        @test minimum(S) < S₀
        # ... and not warm or salinify anywhere.
        @test maximum(T) ≤ T₀ + 1e-6
        @test maximum(S) ≤ S₀ + 1e-6
    end

    @testset "Monin–Obukhov near-wall model" begin
        p = (; drag_coefficient=2.5e-3, minimum_friction_velocity=1e-3,
               heat_transfer_coefficient=0.022, salt_transfer_coefficient=6.2e-4,
               liquidus=DepthDependentLiquidus(), ocean_heat_capacity=3974.0,
               latent_heat=3.34e5, ocean_reference_density=1028.0, ice_density=917.0)
        T, S, z, z_g, speed = 0.5, 34.0, -200.0, 0.5, 0.1

        # Neutral (no buoyancy ⇒ ξ = 0) solve is well-posed and physical
        nwm0 = MoninObukhovNearWall(thermal_expansion=0, haline_contraction=0)
        Tᵦ, Sᵦ, ṁ, u★ = SnowingOcean.solve_interface(nwm0, p, T, S, speed, z, z_g)
        @test isfinite(ṁ) && isfinite(u★)
        @test 0 < Sᵦ < S
        @test Tᵦ ≈ melting_temperature(p.liquidus, Sᵦ, z)
        @test ṁ > 0 && u★ > 0

        # Stable, melt-driven stratification damps near-wall turbulence ⇒ less melt
        nwm = MoninObukhovNearWall()
        _, _, ṁs, _ = SnowingOcean.solve_interface(nwm, p, T, S, speed, z, z_g)
        @test ṁs < ṁ

        # Faster flow ⇒ stronger turbulence ⇒ more melt
        _, _, ṁfast, u★fast = SnowingOcean.solve_interface(nwm, p, T, S, 0.2, z, z_g)
        @test ṁfast > ṁs && u★fast > 0
    end

    @testset "near-wall model immersed boundary condition" begin
        Lz = 8
        grid = RectilinearGrid(size=(8, 8), halo=(3, 3),
                               topology=(Periodic, Flat, Bounded), x=(0, 8), z=(-Lz, 0))
        ice(x, z) = z > -Lz/2
        ibg = ImmersedBoundaryGrid(grid, GridFittedBoundary(ice))

        interface = IceOceanInterface(ibg; formulation=MoninObukhovNearWall())
        bcs = ice_ocean_boundary_conditions(interface)
        model = NonhydrostaticModel(ibg; tracers=(:T, :S),
                                    boundary_conditions=(; T=bcs.T, S=bcs.S, u=bcs.u, v=bcs.v))
        T₀, S₀ = 1.0, 34.0
        set!(model, T=T₀, S=S₀)
        simulation = Simulation(model, Δt=10.0, stop_iteration=5)
        add_melt_flux_callback!(simulation, interface)
        run!(simulation)
        @test minimum(interior(model.tracers.T)) < T₀
        @test minimum(interior(model.tracers.S)) < S₀
        @test all(isfinite, interior(model.tracers.T))
    end

    @testset "frazil model" begin
        fm = FrazilModel(60.0)  # 60 s relaxation timescale
        forcings = frazil_forcing(fm)
        @test haskey(forcings, :T) && haskey(forcings, :ϕ)
        @test haskey(frazil_forcing(fm; frazil_tracer=:frazil), :frazil)

        S, z = 34.0, -10.0
        T★ = melting_temperature(fm.liquidus, S, z)

        # Supercooled water grows frazil and relaxes T upward toward freezing
        FᵀC, FᵠC = SnowingOcean.frazil_tendencies(fm, T★ - 0.05, S, z)
        @test FᵀC > 0 && FᵠC > 0
        # At the freezing point there is no source
        _, Fᵠ0 = SnowingOcean.frazil_tendencies(fm, T★, S, z)
        @test Fᵠ0 ≈ 0 atol=1e-12
        # Warm water melts frazil
        _, FᵠW = SnowingOcean.frazil_tendencies(fm, T★ + 0.05, S, z)
        @test FᵠW < 0
        # The source terms conserve combined sensible + latent energy: c Fᵀ == L Fᵠ
        @test fm.heat_capacity * FᵀC ≈ fm.latent_heat * FᵠC rtol=1e-12
    end
end
