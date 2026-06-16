using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using SnowingOcean
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Printf
#using GLMakie

# Numerical settings and parameters
arch = CPU()
Lx = Ly = 128
Lz = 64
Nx = Ny = 64
Nz = 32
stop_time = 2hours
save_interval = 20minutes
fileprefix = "langmuir_with_teos10"

# Some physical parameters
τˣ = -1e-4
N² = 2e-5
f = 1e-4
h₀ = 10 # initial mixed layer depth
w₀ = 1e-3 # terminal velocity of rising particles
S₀ = 25 # surface salinity
μ = 0.054 # slope of a linear liquidus model
Tᵢ = - μ * S₀ # initial temperature (equal to freezing temperature)

g = gravitational_acceleration = 9.81

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0))

stokes_drift = monochromatic_stokes_drift(amplitude=0.8, wavelength=60,
                                          gravitational_acceleration=g)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τˣ))
coriolis = FPlane(; f) # s⁻¹

c_forcing = rising_tracer_forcing(grid, w₀)

equation_of_state = TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state, gravitational_acceleration)

model = NonhydrostaticModel(grid; coriolis, stokes_drift, buoyancy,
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S, :c),
                            forcing = (; c=c_forcing),
                            boundary_conditions = (; u=u_bcs))

Ξ(z) = randn() * exp(z / 4)

β = SeawaterPolynomials.thermal_expansion(Tᵢ, S₀, 0, equation_of_state)
dz_Sᵢ = - g * β * N²

Sᵢᵢ(z) = if z > - h₀
    S₀
else
    S₀ + dz_Sᵢ * z
end

Δz = zspacings(grid, Center())
ΔS = dz_Sᵢ * model.grid.Lz
Sᵢ(x, y, z) = Sᵢᵢ(z) + 1e-3 * Ξ(z) * ΔS

cᵢ(x, y, z) = if z > - h₀/2
    1
else
    0
end

u★ = sqrt(abs(τˣ))
uᵢ(x, y, z) = u★ * 1e-3 * Ξ(z)

set!(model, u=uᵢ, w=uᵢ, S=Sᵢ, T=Tᵢ, c=cᵢ)

simulation = Simulation(model; Δt=1.0, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=1minute)

function progress(simulation)
    u, v, w = simulation.model.velocities

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, max|u| = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, interior(u)),
                   maximum(abs, interior(v)),
                   maximum(abs, interior(w)),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

Nz = size(grid, 3)
outputs = merge(model.velocities, model.tracers)
simulation.output_writers[:xy] = JLD2Writer(model, outputs,
                                            indices = (:, :, Nz),
                                            schedule = TimeInterval(save_interval),
                                            filename = fileprefix * "_xy.jld2",
                                            overwrite_existing = true)

simulation.output_writers[:yz] = JLD2Writer(model, outputs,
                                            indices = (1, :, :),
                                            schedule = TimeInterval(save_interval),
                                            filename = fileprefix * "_yz.jld2",
                                            overwrite_existing = true)

averages = (
    u =  Average(model.velocities.u, dims=(1, 2)),
    v =  Average(model.velocities.v, dims=(1, 2)),
    S =  Average(model.tracers.S, dims=(1, 2)),
    T =  Average(model.tracers.T, dims=(1, 2)),
    c =  Average(model.tracers.c, dims=(1, 2)),
)

simulation.output_writers[:z] = JLD2Writer(model, averages,
                                           schedule = TimeInterval(save_interval),
                                           filename = fileprefix * "_averages.jld2",
                                           overwrite_existing = true)

run!(simulation)

#=
wt = FieldTimeSeries("langmuir_particles_xy.jld2", "w")
cxyt = FieldTimeSeries("langmuir_particles_xy.jld2", "c")
uyzt = FieldTimeSeries("langmuir_particles_yz.jld2", "u")
cyzt = FieldTimeSeries("langmuir_particles_yz.jld2", "c")
Nt = length(wt)

fig = Figure(size=(1200, 600))
axw = Axis(fig[1, 1])
axcxy = Axis(fig[1, 2])
axuyz = Axis(fig[2, 1])
axcyz = Axis(fig[2, 2])
slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value
wn = @lift interior(wt[$n], :, :, 1)
cxyn = @lift interior(cxyt[$n], :, :, 1)
uyzn = @lift interior(uyzt[$n], 1, :, :)
cyzn = @lift interior(cyzt[$n], 1, :, :)

wlim = 1e-2
clim = 5e-2

heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)
heatmap!(axcxy, cxyn, colorrange=(0, clim), colormap=:solar)

heatmap!(axuyz, uyzn, colorrange=(-wlim, wlim), colormap=:balance)
heatmap!(axcyz, cyzn, colorrange=(0, clim), colormap=:solar)

display(fig)
=#
