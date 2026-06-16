using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using SnowingOcean
using Printf
#using GLMakie

arch = GPU()
Lx = Ly = 128
Lz = 64
Nx = Ny = 128
Nz = 64

# Boundary conditions
τˣ = -1e-4
N² = 2e-5
f = 1e-4
h₀ = 10 # initial mixed layer depth
w₀ = 1e-3 # terminal velocity of rising particles

grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), halo=(5, 5, 5), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

stokes_drift = monochromatic_stokes_drift(amplitude=0.8, wavelength=60)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τˣ))
coriolis = FPlane(; f) # s⁻¹

c_forcing = rising_tracer_forcing(grid, w₀)

model = NonhydrostaticModel(grid; coriolis, stokes_drift,
                            advection = WENO(order=9),
                            timestepper = :RungeKutta3,
                            tracers = (:b, :c),
                            buoyancy = BuoyancyTracer(),
                            forcing = (; c=c_forcing),
                            boundary_conditions = (; u=u_bcs))

Ξ(z) = randn() * exp(z / 4)

bᴺ(z) = if z > - h₀
    - N² * h₀
else
    N² * z
end

Δz = zspacings(grid, Center())

bᵢ(x, y, z) = bᴺ(z) + 1e-3 * Ξ(z) * N² * model.grid.Lz

cᵢ(x, y, z) = if z > - h₀/2
    1
else
    0
end

u★ = sqrt(abs(τˣ))
uᵢ(x, y, z) = u★ * 1e-3 * Ξ(z)

set!(model, u=uᵢ, w=uᵢ, b=bᵢ, c=cᵢ)

simulation = Simulation(model, Δt=20.0, stop_time = 8hours)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=1minute)

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

time_interval = 20minutes
Nz = size(grid, 3)
outputs = merge(model.velocities, model.tracers)
simulation.output_writers[:xy] = JLD2Writer(model, outputs,
                                            indices = (:, :, Nz),
                                            schedule = TimeInterval(time_interval),
                                            filename = "langmuir_particles_xy.jld2",
                                            overwrite_existing = true)

simulation.output_writers[:yz] = JLD2Writer(model, outputs,
                                            indices = (1, :, :),
                                            schedule = TimeInterval(time_interval),
                                            filename = "langmuir_particles_yz.jld2",
                                            overwrite_existing = true)

averages = (
    u =  Average(model.velocities.u, dims=(1, 2)),
    v =  Average(model.velocities.v, dims=(1, 2)),
    b =  Average(model.tracers.b, dims=(1, 2)),
    c =  Average(model.tracers.c, dims=(1, 2)),
)

simulation.output_writers[:z] = JLD2Writer(model, averages,
                                           schedule = TimeInterval(time_interval),
                                           filename = "langmuir_particles_averages.jld2",
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
