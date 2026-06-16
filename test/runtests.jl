using SnowingOcean
using Oceananigans
using Test

@testset "SnowingOcean" begin

    @testset "LinearLiquidus" begin
        liquidus = LinearLiquidus()
        # Freezing point decreases with salinity and with depth
        @test melting_temperature(liquidus, 0, 0) ≈ liquidus.reference_temperature
        @test melting_temperature(liquidus, 35, 0) < melting_temperature(liquidus, 30, 0)
        @test melting_temperature(liquidus, 34, -1000) < melting_temperature(liquidus, 34, 0)
        # ~0.76 °C depression at 1000 m for the default depth slope
        ΔT = melting_temperature(liquidus, 34, 0) - melting_temperature(liquidus, 34, -1000)
        @test ΔT ≈ 0.761 rtol=1e-3
    end

    @testset "three-equation interface solve" begin
        liquidus = LinearLiquidus()
        Γ_T, Γ_S = 0.022, 6.2e-4
        c, L, ρₒ, ρᵢ = 3974.0, 3.34e5, 1028.0, 917.0
        u★ = 0.01
        S, z = 34.0, -200.0
        three = Val(:three_equation)

        # Warm ocean ⇒ melting: 0 < Sᵦ < S, interface at freezing point, ṁ > 0
        T = 1.0
        Tᵦ, Sᵦ, ṁ = interface_temperature_and_salinity(three, liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test 0 < Sᵦ < S
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, z)
        @test ṁ > 0
        @test Tᵦ < T  # interface colder than warm ocean

        # Ocean exactly at the freezing point ⇒ no melt
        Tf = melting_temperature(liquidus, S, z)
        _, _, ṁ0 = interface_temperature_and_salinity(three, liquidus, Tf, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test abs(ṁ0) < 1e-10

        # Supercooled ocean ⇒ freezing (ṁ < 0)
        _, _, ṁf = interface_temperature_and_salinity(three, liquidus, Tf - 0.1, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test ṁf < 0

        # Stronger thermal driving ⇒ more melt
        _, _, ṁ2 = interface_temperature_and_salinity(three, liquidus, 2.0, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test ṁ2 > ṁ

        # Two-equation: interface salinity is the far-field salinity
        Tᵦ2, Sᵦ2, ṁ2eq = interface_temperature_and_salinity(Val(:two_equation), liquidus, T, S, z, u★, Γ_T, Γ_S, c, L, ρₒ, ρᵢ)
        @test Sᵦ2 == S
        @test Tᵦ2 ≈ melting_temperature(liquidus, S, z)
        @test ṁ2eq > 0
    end

    @testset "ThreeEquationInterface construction" begin
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        interface = ThreeEquationInterface(grid)
        @test interface.temperature_flux isa Field
        @test interface.salt_flux isa Field
        @test all(interior(interface.temperature_flux) .== 0)
        @test all(interior(interface.salt_flux) .== 0)
        @test_throws ArgumentError ThreeEquationInterface(grid; formulation=:bogus)
    end

    @testset "compute_melt_fluxes! fills the flux fields" begin
        grid = RectilinearGrid(size=(4, 1, 8), halo=(5, 1, 5),
                               topology=(Periodic, Periodic, Bounded), extent=(10, 10, 80))
        interface = ThreeEquationInterface(grid; minimum_friction_velocity=0.01)
        model = NonhydrostaticModel(grid; tracers=(:T, :S))
        set!(model, T=1.0, S=34.0)  # warm, above freezing
        compute_melt_fluxes!(model, interface)
        Jᵀ = interior(interface.temperature_flux)
        Jˢ = interior(interface.salt_flux)
        @test all(Jᵀ .> 0)   # warm ocean ⇒ heat leaves the ocean (melting)
        @test all(Jˢ .> 0)   # meltwater freshens ⇒ salt leaves the ocean
    end

    @testset "immersed melt boundary condition cools and freshens the ocean" begin
        Lz = 8
        grid = RectilinearGrid(size=(4, 8), halo=(5, 5),
                               topology=(Periodic, Flat, Bounded),
                               x=(0, 4), z=(-Lz, 0))

        # Flat ice slab occupying the upper half of the domain
        ice(x, z) = z > -Lz/2
        ibg = ImmersedBoundaryGrid(grid, GridFittedBoundary(ice))

        interface = ThreeEquationInterface(ibg; minimum_friction_velocity=0.05)
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
        # Melting of warm water should cool and freshen the ocean adjacent to the ice
        @test minimum(T) < T₀
        @test minimum(S) < S₀
        # ... and should not warm or salinify anywhere
        @test maximum(T) ≤ T₀ + 1e-6
        @test maximum(S) ≤ S₀ + 1e-6
    end
end
