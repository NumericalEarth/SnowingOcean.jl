# # Melting beneath an ice lid (2D)
#
# This example illustrates the `SnowingOcean` ice–ocean melt interface on a simple
# two-dimensional (``x``–``z``) problem: a slab of ice floating over warm, salty water.
# Heat carried to the ice base melts it; the cold, fresh meltwater forms a buoyant
# boundary layer beneath the ice. It is a stripped-down version of the ice-shelf cavity
# setup — flat ice, no sill, no rotation — meant to show the user interface.
#
# ## The melt parameterization
#
# At the ice–ocean interface the temperature ``T_b``, salinity ``S_b`` and melt rate
# ``\dot m`` (m s⁻¹, positive for melting) are determined by a turbulent heat balance, a
# salt balance, and the requirement that the interface sits at the local freezing point
# (the **three-equation** formulation of Holland & Jenkins, 1999):
#
# ```math
# \begin{aligned}
# \rho_o c_o \, \gamma_T \,(T - T_b) &= \rho_i L \, \dot m , \\
# \rho_o \, \gamma_S \,(S - S_b)     &= \rho_i \, \dot m \, S_b , \\
# T_b &= T^f(S_b, z) = T_0 - \Gamma S_b + \lambda z .
# \end{aligned}
# ```
#
# Here ``\gamma_T = \Gamma_T u_\star`` and ``\gamma_S = \Gamma_S u_\star`` are turbulent
# exchange velocities, ``u_\star = \sqrt{C_d \, |\mathbf{u}|^2}`` is the friction velocity,
# and ``T^f`` is the (depth-dependent, linear) freezing temperature. Because the liquidus
# is linear, the system reduces to a **closed-form quadratic** for ``S_b`` — no iteration.
# The **two-equation** formulation ([`TwoEquation`](@ref)) instead sets ``S_b = S`` and
# obtains ``\dot m`` directly from the heat balance.
#
# `SnowingOcean` computes the kinematic boundary fluxes ``J^T = -\gamma_T (T - T_b)`` and
# ``J^S = -\gamma_S (S - S_b)`` into `Field`s (in a callback), and applies them as flux
# boundary conditions on the immersed ice base — with the *same* drag coefficient ``C_d``
# used for the momentum `BulkDrag`, so the melt and momentum fluxes are consistent.

using SnowingOcean
using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CairoMakie

# ## Grid and immersed ice lid
#
# A 2D `x`–`z` domain (`Flat` in `y`). The ice occupies the top `ice_thickness` meters;
# everything below is ocean. Increasing the `y` size turns this into a 3D simulation.

Lx, Lz = 64, 32
Nx, Nz = 96, 48
ice_thickness = 4   # m

grid = RectilinearGrid(size = (Nx, Nz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded),
                       x = (0, Lx),
                       z = (-Lz, 0))

@inline is_ice(x, z) = z > -ice_thickness
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(is_ice))

# ## The ice–ocean interface and boundary conditions
#
# `IceOceanInterface` allocates the flux `Field`s and stores the melt/drag parameters;
# `ice_ocean_boundary_conditions` builds the `(u, v, T, S)` boundary conditions that read
# them on the immersed ice base. The default `formulation = ThreeEquation()` uses constant
# transfer coefficients; pass `formulation = MoninObukhovNearWall()` instead for the
# wall-model closure of Vreugdenhil et al. (2022), which computes the friction velocity and
# transfer coefficients self-consistently with the near-wall stratification.

interface = IceOceanInterface(grid)
bcs = ice_ocean_boundary_conditions(interface)

# ## Model
#
# A non-hydrostatic model with a linear equation of state, so the cold/fresh meltwater
# layer is buoyant relative to the warm, salty interior. We use a
# `ConjugateGradientPoissonSolver`: the default FFT-based pressure solver is only
# approximate next to immersed boundaries and would leave the velocity divergent at the
# ice base, so it is the appropriate choice whenever the ice is an immersed boundary.

equation_of_state = LinearEquationOfState(thermal_expansion=3.87e-5, haline_contraction=7.86e-4)
buoyancy = SeawaterBuoyancy(; equation_of_state)

model = NonhydrostaticModel(grid; buoyancy,
                            advection = WENO(order=9),
                            tracers = (:T, :S),
                            closure = ScalarDiffusivity(ν=1e-3, κ=1e-3),
                            pressure_solver = ConjugateGradientPoissonSolver(grid),
                            boundary_conditions = (; u=bcs.u, v=bcs.v, T=bcs.T, S=bcs.S))

# Warm, salty water at rest, with a touch of noise to break symmetry.

Tᵢ(x, z) = 0.5 + 1e-3 * randn()
set!(model, T=Tᵢ, S=34)

# ## Run
#
# A short simulation. The melt-flux boundary conditions update themselves inside
# `update_state!` each step — no callback is needed.

simulation = Simulation(model, Δt=1.0, stop_time=20minutes)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=10.0)

run!(simulation)

# ## Visualize
#
# Temperature and salinity at the end of the run. The ocean cools and freshens in a
# boundary layer beneath the ice (shown in gray).

xc = xnodes(grid, Center())
zc = znodes(grid, Center())

solid = [is_ice(x, z) for x in xc, z in zc]
mask(φ) = (ψ = Array{Float64}(interior(φ)[:, 1, :]); ψ[solid] .= NaN; ψ)
function nanrange(ψ)
    lo, hi = extrema(filter(!isnan, ψ))
    return lo == hi ? (lo - 1, hi + 1) : (lo, hi)
end

Tn = mask(model.tracers.T)
Sn = mask(model.tracers.S)

fig = Figure(size=(900, 700))
axT = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="Temperature (°C)")
axS = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="Salinity (g kg⁻¹)")

hmT = heatmap!(axT, xc, zc, Tn; colormap=:thermal, colorrange=nanrange(Tn), nan_color=:gray70)
hmS = heatmap!(axS, xc, zc, Sn; colormap=:haline, colorrange=nanrange(Sn), nan_color=:gray70)
Colorbar(fig[1, 2], hmT)
Colorbar(fig[2, 2], hmS)

fig
