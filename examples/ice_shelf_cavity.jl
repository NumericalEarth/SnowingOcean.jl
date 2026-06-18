# # Ice-shelf cavity geometry for a NonhydrostaticModel
#
# This script builds an `ImmersedBoundaryGrid` whose geometry resembles an
# Antarctic ice-shelf cavity (see Fig. 3 of Hoppmann et al. 2020, "Platelet ice,
# the Southern Ocean's hidden ice"):
#
#   * a floating ice shelf on the left whose base slopes down from a shallow
#     draft at the ice front to the grounding line, where it meets the seafloor;
#   * open ocean on the right (a fraction `ocean_fraction` of the domain);
#   * a flat seafloor with an optional Gaussian sill near the ice front that
#     partially isolates the cavity from the open ocean.
#
# The bathymetry is fully parameterized (see `bathymetry` below) so it is easy
# to tweak. We start in 2D (`Ny = 1`, flow into the page along `y`); increasing
# `Ny` turns this into a 3D turbulence-resolving simulation with no other
# changes. For now we only build the grid, the immersed boundary, and the model,
# then visualize the initial condition.

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

# ## Domain and resolution
#
# Set `Ny = 1` for a 2D (x–z) simulation with a flow into the page; increase
# `Ny` (and it switches to a periodic, turbulence-resolving y direction).

arch = CPU()

Lx = 4000     # domain length (m)
Ly = 500      # domain width (m), only used when Ny > 1
Lz = 500      # domain depth (m)

Nx = 256
Ny = 1
Nz = 64

# ## Bathymetry parameters
#
# All lengths in meters; `z = 0` is the ocean surface and `z = -Lz` the deepest
# seafloor. Ice-base drafts are negative (below the surface).

bathymetry = (; Lx, Lz,
              ocean_fraction   = 1/3,         # fraction of the domain that is open ocean (right)
              draft_front      = -100,        # ice-shelf base depth at the ice front
              draft_gl         = -Lz,         # ice-shelf base depth at the grounding line (meets the bed)
              draft_power      = 1.0,         # 1 → linear base; >1 → concave (deeper near grounding line)
              seafloor_step_center = 2 * Lx / 3, # x-position of the seafloor slope (near the ice front)
              seafloor_step_height = 50,      # rise of the seafloor from the deep cavity to the shallow ocean
              seafloor_step_width  = 200)     # width of the seafloor slope (smaller ⟹ steeper)

# Ice-front position: cavity is to the left, open ocean to the right.
x_front(p) = p.Lx * (1 - p.ocean_fraction)

# Ice-shelf base draft z = ice_draft(x) ≤ 0. In the open ocean (x ≥ x_front)
# there is no ice and the surface is at z = 0.
@inline function ice_draft(x, p)
    xf = x_front(p)
    if x ≥ xf
        return zero(x)
    else
        ξ = (xf - x) / xf  # 0 at the ice front, 1 at the grounding line (x = 0)
        return p.draft_front + (p.draft_gl - p.draft_front) * ξ^p.draft_power
    end
end

# Seafloor height z = seafloor(x) ≤ 0: a tanh step that is deep (-Lz) in the
# cavity on the left and shallow (-Lz + step_height) in the open ocean on the
# right, with the slope centered at `seafloor_step_center`.
@inline function seafloor(x, p)
    step = (p.seafloor_step_height / 2) * (1 + tanh((x - p.seafloor_step_center) / p.seafloor_step_width))
    return -p.Lz + step
end

# A cell is solid (immersed) if it lies above the ice-shelf base or below the
# seafloor. The geometry is independent of y.
@inline is_immersed(x, z, p) = (z > ice_draft(x, p)) | (z < seafloor(x, p))

# ## Grid and immersed boundary
#
# `Flat` y in 2D (so functions are called with `(x, z)`); `Periodic` y in 3D.

if Ny == 1
    underlying_grid = RectilinearGrid(arch,
                                      topology = (Bounded, Flat, Bounded),
                                      size = (Nx, Nz),
                                      halo = (5, 5),
                                      x = (0, Lx),
                                      z = (-Lz, 0))

    mask(x, z) = is_immersed(x, z, bathymetry)
else
    underlying_grid = RectilinearGrid(arch,
                                      topology = (Bounded, Periodic, Bounded),
                                      size = (Nx, Ny, Nz),
                                      halo = (5, 5, 5),
                                      x = (0, Lx),
                                      y = (0, Ly),
                                      z = (-Lz, 0))

    mask(x, y, z) = is_immersed(x, z, bathymetry)
end

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(mask))

@info "Built immersed-boundary grid:" grid

# ## Model
#
# A `NonhydrostaticModel` with a single buoyancy tracer, WENO(order=9)
# advection, and an f-plane. We use the default FFT-based pressure solver for
# now; note it is only approximate next to immersed boundaries, so once we start
# time-stepping we may want to switch to a `ConjugateGradientPoissonSolver`.
# Boundary conditions at the ice base and seafloor (melting/freezing fluxes)
# can be added later.

f = -1.3e-4  # Southern Ocean Coriolis parameter (s⁻¹)

model = NonhydrostaticModel(grid;
                            advection = WENO(order=9),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(; f))

# ## Initial condition
#
# A stable linear buoyancy stratification and a uniform flow into the page
# (along +y). The `b` initializer takes `(x, z)` in 2D and `(x, y, z)` in 3D.

N² = 1e-5  # buoyancy frequency squared (s⁻²)
V₀ = 0.05  # along-cavity inflow speed, into the page (m s⁻¹)

bᵢ = Ny == 1 ? (x, z) -> N² * z : (x, y, z) -> N² * z

set!(model, b=bᵢ, v=V₀)

# ## Visualize the initial condition (x–z cross section)
#
# We plot the `b` field directly: the Oceananigans Makie extension supplies the `x`–`z`
# coordinates and masks the immersed ice and bedrock (shown in gray). We overlay the
# ice-shelf base and seafloor, filled to distinguish ice (blue) from rock (brown).

xf = range(0, Lx, length=600)
draft_line = ice_draft.(xf, Ref(bathymetry))
floor_line = seafloor.(xf, Ref(bathymetry))

fig = Figure(size=(1100, 450))
ax = Axis(fig[1, 1];
          xlabel = "x (m)",
          ylabel = "z (m)",
          title = @sprintf("Ice-shelf cavity initial condition (flow into page, v = %.2f m s⁻¹)", V₀))

hm = heatmap!(ax, model.tracers.b; colormap=:deep, nan_color=:gray70)

# Ice shelf (fill from the base up to the surface) and bedrock (fill below the seafloor)
band!(ax, xf, draft_line, fill(0.0, length(xf)); color=(:skyblue, 0.35))
band!(ax, xf, fill(-Lz, length(xf)), floor_line; color=(:saddlebrown, 0.55))

lines!(ax, xf, draft_line; color=:black, linewidth=2)
lines!(ax, xf, floor_line; color=:saddlebrown, linewidth=2)
vlines!(ax, [x_front(bathymetry)]; color=:black, linestyle=:dash, linewidth=1)

xlims!(ax, 0, Lx)
ylims!(ax, -Lz, 0)
Colorbar(fig[1, 2], hm; label="buoyancy b (m s⁻²)")

output = joinpath(@__DIR__, "ice_shelf_cavity_initial.png")
save(output, fig)
@info "Saved initial-condition figure to $output"

fig
