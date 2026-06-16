using GLMakie
using GibbsSeaWater
using Printf

fig = Figure(size=(1200, 400))
axp = Axis(fig[1, 1], xlabel="Depth (m)", ylabel="Freezing temperature (ᵒC)")
axS = Axis(fig[1, 2], xlabel="Absolute salinity (g kg⁻¹)", ylabel="Freezing temperature (ᵒC)")
axa = Axis(fig[1, 3], xlabel="Absolute salinity (g kg⁻¹)", ylabel="Thermal expansion \n at freezing temperature (kg m⁻³ ᵒC⁻¹")

ϵ = 1e-4 # conversion from Pa to dbar
g = 9.81
ρ₀ = 1035 # reference density
p₀ = 10.1325 # sealevel pressure in dbar
d = 0.0:10.0:1000 # depth
pressure(d) = p₀ + ρ₀ * g * d * ϵ
p = pressure.(d)

for Sᴬ = (0, 20, 30, 40)
    T = GibbsSeaWater.gsw_ct_freezing.(Sᴬ, p, 0)
    lbl = @sprintf("Sᴬ = %d g/kg", Sᴬ)
    lines!(axp, d, T, label=lbl)

    dTdz = (T[end] - T[1]) / (d[end] - d[1])
    @show dTdz
end
    
Sᴬ = 0.0:0.1:50.0

for d = (0, 100, 1000)
    p = pressure(d)
    T = GibbsSeaWater.gsw_ct_freezing.(Sᴬ, p + p₀, 0)
    lbl = @sprintf("z = -%d m", d)
    lines!(axS, Sᴬ, T, label=lbl)
end

for d = (0, 100, 1000)
    p = pressure(d)
    T★ = GibbsSeaWater.gsw_ct_freezing.(Sᴬ, p + p₀, 0)
    α = gsw_alpha.(Sᴬ, T★, p + p₀)
    lbl = @sprintf("z = -%d m", d)
    lines!(axa, Sᴬ, α, label=lbl)
end
 
axislegend(axp, position=:rt)
axislegend(axS, position=:lb)
axislegend(axa, position=:rb)

display(fig)

