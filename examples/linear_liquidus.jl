using GLMakie
using GibbsSeaWater
using SnowingOcean
using Printf

fig = Figure(size=(600, 600))
axp = Axis(fig[1, 1], xlabel="Depth (m)", ylabel="Freezing temperature (ᵒC)")

ϵ = 1e-4 # conversion from Pa to dbar
g = 9.81
ρ₀ = 1035 # reference density
p₀ = 10.1325 # sealevel pressure in dbar
d = 0.0:10.0:1000 # depth
pressure(d) = p₀ + ρ₀ * g * d * ϵ
p = pressure.(d)

μ = 0.0545
γ = 7.9e-4

for Sᴬ = (0, 20, 30, 40)
    Texact = GibbsSeaWater.gsw_ct_freezing.(Sᴬ, p, 0)
    z = @. -d
    Tapprox = linear_freezing_temperature.(Sᴬ, z; slope=μ, depth_coefficient=γ)
    lbl = @sprintf("Sᴬ = %d g/kg", Sᴬ)
    lines!(axp, d, Texact, linewidth=4, label=lbl)
    lines!(axp, d, Tapprox, linestyle=:dash)
end

axislegend(axp)
display(fig)


