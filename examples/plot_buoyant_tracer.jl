using Oceananigans
using GLMakie

wt = FieldTimeSeries("langmuir_particles_xy.jld2", "w")
cxyt = FieldTimeSeries("langmuir_particles_xy.jld2", "c")
uyzt = FieldTimeSeries("langmuir_particles_yz.jld2", "u")
cyzt = FieldTimeSeries("langmuir_particles_yz.jld2", "c")
Nt = length(wt)

fig = Figure(size=(1200, 800))
axw = Axis(fig[1, 1], aspect=1)
axcxy = Axis(fig[1, 2], aspect=1)
axuyz = Axis(fig[2, 1], aspect=2)
axcyz = Axis(fig[2, 2], aspect=2)
slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value
wn = @lift interior(wt[$n], :, :, 1)
cxyn = @lift interior(cxyt[$n], :, :, 1)
uyzn = @lift interior(uyzt[$n], 1, :, :)
cyzn = @lift interior(cyzt[$n], 1, :, :)

ulim = 1e-1
wlim = 1e-2
clim = 1.0

heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)
heatmap!(axcxy, cxyn, colorrange=(0, clim), colormap=:solar)

heatmap!(axuyz, uyzn, colorrange=(-ulim, ulim), colormap=:balance)
heatmap!(axcyz, cyzn, colorrange=(0, clim), colormap=:solar)

display(fig)
