import openmc
import os

os.makedirs("output", exist_ok=True)
os.chdir("output")

##############################################
                # Materials #
##############################################

uo2 = openmc.Material(name="uo2")
uo2.add_nuclide('U235',0.04)
uo2.add_nuclide('U238',0.96)
uo2.add_nuclide('O16',2.0)
uo2.set_density('g/cm3',10.4)
uo2.depletable = True

zirconium = openmc.Material(name="zirconium")
zirconium.add_element('Zr',1.0)
zirconium.set_density('g/cm3',6.55)
zirconium.depletable = False

steel = openmc.Material(name='Stainless Steel')
steel.add_element('C', 0.08, percent_type='wo')
steel.add_element('Si', 1.00, percent_type='wo')
steel.add_element('P', 0.045, percent_type='wo')
steel.add_element('S', 0.030, percent_type='wo')
steel.add_element('Mn', 2.00, percent_type='wo')
steel.add_element('Cr', 20.0, percent_type='wo')
steel.add_element('Ni', 11.0, percent_type='wo')
steel.add_element('Fe', 65.845, percent_type='wo')
steel.set_density('g/cm3', 8.00)

water = openmc.Material(name="water")
water.add_nuclide('H1',2.0)
water.add_nuclide('O16',1.0)
water.set_density('g/cm3',1.0)
water.add_s_alpha_beta('c_H_in_H2O')

b4c = openmc.Material(name='b4c')
b4c.add_element('B', 4.0)
b4c.add_element('C', 1.0)
b4c.set_density('g/cm3', 2.52)


materials = openmc.Materials([uo2, zirconium, water, steel, b4c])
materials.export_to_xml()

##############################################
                # Geometry #
##############################################

########## Fuel Pins ##########

bottom = openmc.ZPlane(z0=0.0, boundary_type='reflective')
z1 = openmc.ZPlane(z0=40.0)
z2 = openmc.ZPlane(z0=80.0)
top = openmc.ZPlane(z0=120.0, boundary_type='reflective')

fuel_outer_radius = openmc.ZCylinder(r=0.4096)
clad_inner_radius = openmc.ZCylinder(r=0.4179)
clad_outer_radius = openmc.ZCylinder(r=0.4750)

fuel_region = -fuel_outer_radius 
gap_region = +fuel_outer_radius & -clad_inner_radius 
clad_region = +clad_inner_radius & -clad_outer_radius 

fuel_cell = openmc.Cell(1, name='fuel')
fuel_cell.fill = uo2
fuel_cell.region = fuel_region

gap_cell = openmc.Cell(name='air gap')
gap_cell.region = gap_region

clad_cell = openmc.Cell(name='clad')
clad_cell.fill = zirconium
clad_cell.region = clad_region

pitch = 1.26

box = openmc.model.RectangularPrism(width=pitch, height=pitch,
                                    boundary_type='reflective')
type(box)

fuel_water_region = +clad_outer_radius & -box 

fuel_water_cell = openmc.Cell(name='fuel water')
fuel_water_cell.fill = water
fuel_water_cell.region = fuel_water_region

fuel_universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, fuel_water_cell])

########## Guide Pins ##########

waterrod_outer_radius = openmc.ZCylinder(r=0.4179)
wr_clad_outer_radius = openmc.ZCylinder(r=0.4750)

waterrod_inner_region = -waterrod_outer_radius 
wr_clad_region = +waterrod_outer_radius & -wr_clad_outer_radius 
waterrod_outer_region = +wr_clad_outer_radius & -box 

waterrod_inner_cell = openmc.Cell(name='waterrod_inner')
waterrod_inner_cell.fill = water
waterrod_inner_cell.region = waterrod_inner_region

wr_clad_cell = openmc.Cell(name='wr_clad')
wr_clad_cell.fill = zirconium
wr_clad_cell.region = wr_clad_region

waterrod_outer_cell = openmc.Cell(name='waterrod_outer')
waterrod_outer_cell.fill = water
waterrod_outer_cell.region = waterrod_outer_region

mod_universe = openmc.Universe(cells=[waterrod_inner_cell, wr_clad_cell, waterrod_outer_cell])

########## Instrument Pin ##########

ipin_inner_radius = openmc.ZCylinder(r=0.4750)
ipin_outer_radius = openmc.ZCylinder(r=0.5500)

ipin_steel = -ipin_inner_radius & +bottom & -top
ipin_clad = +ipin_inner_radius & -ipin_outer_radius & +bottom & -top
ipin_water = +ipin_outer_radius & -box & +bottom & -top

ipin_steel_cell = openmc.Cell(name='ipin_rod')
ipin_steel_cell.fill = steel
ipin_steel_cell.region = ipin_steel

ipin_clad_cell = openmc.Cell(name='ipin_clad')
ipin_clad_cell.fill = zirconium
ipin_clad_cell.region = ipin_clad

ipin_water_cell = openmc.Cell(name='ipin_water')
ipin_water_cell.fill = water
ipin_water_cell.region = ipin_water

ipin_universe = openmc.Universe(cells=[ipin_steel_cell, ipin_clad_cell, ipin_water_cell])

######### Control Rods ##########

abs_outer_radius = openmc.ZCylinder(r=0.4)
clad_outer_radius = openmc.ZCylinder(r=0.46)

abs_region = -abs_outer_radius 
abs_clad_region = +abs_outer_radius & -clad_outer_radius
abs_water_region = +clad_outer_radius & -box

absorber = openmc.Cell(name='absorber')
absorber.fill = b4c
absorber.region = abs_region

abs_clad = openmc.Cell(name='absorber cladding')
abs_clad.fill = zirconium
abs_clad.region = abs_clad_region

abs_water = openmc.Cell(name='absorber water')
abs_water.fill = water
abs_water.region = abs_water_region

absorber_universe = openmc.Universe(cells=[absorber, abs_clad, abs_water])

##############################################
                # Assembly #
##############################################

########## Control Rod Insertions ##########

## No Rods ##

bottom_cell_00 = openmc.Cell()
bottom_cell_00.region = +bottom & -z1
bottom_cell_00.fill = mod_universe

middle_cell_00 = openmc.Cell()
middle_cell_00.region = +z1 & -z2
middle_cell_00.fill = mod_universe  

top_cell_00 = openmc.Cell()
top_cell_00.region = +z2 & -top
top_cell_00.fill = mod_universe

## One Third ##

bottom_cell_33 = openmc.Cell()
bottom_cell_33.region = +bottom & -z1
bottom_cell_33.fill = absorber_universe

middle_cell_33 = openmc.Cell()
middle_cell_33.region = +z1 & -z2
middle_cell_33.fill = mod_universe  

top_cell_33 = openmc.Cell()
top_cell_33.region = +z2 & -top
top_cell_33.fill = mod_universe

## Two Thirds ##

bottom_cell_66 = openmc.Cell()
bottom_cell_66.region = +bottom & -z1
bottom_cell_66.fill = absorber_universe

middle_cell_66 = openmc.Cell()
middle_cell_66.region = +z1 & -z2
middle_cell_66.fill = absorber_universe 

top_cell_66 = openmc.Cell()
top_cell_66.region = +z2 & -top
top_cell_66.fill = mod_universe

## Full Insertion ##

bottom_cell_100 = openmc.Cell()
bottom_cell_100.region = +bottom & -z1
bottom_cell_100.fill = absorber_universe

middle_cell_100 = openmc.Cell()
middle_cell_100.region = +z1 & -z2
middle_cell_100.fill = absorber_universe  

top_cell_100 = openmc.Cell()
top_cell_100.region = +z2 & -top
top_cell_100.fill = absorber_universe


universe_0 = openmc.Universe(cells=[bottom_cell_00, middle_cell_00, top_cell_00])
universe_33 = openmc.Universe(cells=[bottom_cell_33, middle_cell_33, top_cell_33])
universe_66 = openmc.Universe(cells=[bottom_cell_66, middle_cell_66, top_cell_66])
universe_100 = openmc.Universe(cells=[bottom_cell_100, middle_cell_100, top_cell_100])

########## Assembly ##########

import numpy as np

def create_assembly(fuel_universe, instrument_universe, guide_universe, pitch, name):
    layout = np.full((17,17), 'F')
    guide_positions = [
        (0,0),(0,4),(0,8),(0,12),(0,16),
        (4,0),(4,4),(4,8),(4,12),(4,16),
        (8,0),(8,4),(8,12),(8,16),
        (12,0),(12,4),(12,8),(12,12),(12,16),
        (16,0),(16,4),(16,8),(16,12),(16,16),
    ]
    
    for i, j in guide_positions:
        layout[i,j] = 'G'

    layout[8,8] = 'I'

    universe_map = {'F': fuel_universe, 'G': guide_universe, 'I': instrument_universe}
    universes = [[universe_map[cell] for cell in row] for row in layout]

    full_pitch = pitch * 17
    lattice = openmc.RectLattice(name='UO2 Assembly')
    lattice.pitch = (pitch,pitch)
    lattice.lower_left = [-full_pitch/2, -full_pitch/2]
    lattice.universes = universes
    return lattice

uo2_assembly_0 = create_assembly(fuel_universe, ipin_universe, universe_0, pitch, 'UO2 Assembly - No Rods')
uo2_assembly_33 = create_assembly(fuel_universe, ipin_universe, universe_33, pitch, 'UO2 Assembly - One Third')
uo2_assembly_66 = create_assembly(fuel_universe, ipin_universe, universe_66, pitch, 'UO2 Assembly - Two Thirds')
uo2_assembly_100 = create_assembly(fuel_universe, ipin_universe, universe_100, pitch, 'UO2 Assembly - Full Rods')

########## Sleeve and Outer Water ##########

full_pitch = pitch * 17
sleeve_thickness = 0.1

assembly_region = -openmc.model.RectangularPrism(width=full_pitch, height=full_pitch, origin=(0,0)) & +bottom & -top

fuel_assembly_cell_0 = openmc.Cell(name='fuel assembly no rods', fill=uo2_assembly_0, region=assembly_region)
fuel_assembly_cell_33 = openmc.Cell(name='fuel assembly one third', fill=uo2_assembly_33, region=assembly_region)
fuel_assembly_cell_66 = openmc.Cell(name='fuel assembly two thirds', fill=uo2_assembly_66, region=assembly_region)
fuel_assembly_cell_100 = openmc.Cell(name='fuel assembly full rods', fill=uo2_assembly_100, region=assembly_region)

assembly_sleeve_00 = openmc.Cell(name='full assembly sleeve')
assembly_sleeve_00.region = -openmc.model.RectangularPrism(width=full_pitch+2*sleeve_thickness, height=full_pitch+2*sleeve_thickness) & ~assembly_region & +bottom & -top
assembly_sleeve_00.fill = zirconium

assembly_sleeve_33 = openmc.Cell(name='full assembly sleeve')
assembly_sleeve_33.region = -openmc.model.RectangularPrism(width=full_pitch+2*sleeve_thickness, height=full_pitch+2*sleeve_thickness) & ~assembly_region & +bottom & -top
assembly_sleeve_33.fill = zirconium

assembly_sleeve_66 = openmc.Cell(name='full assembly sleeve')
assembly_sleeve_66.region = -openmc.model.RectangularPrism(width=full_pitch+2*sleeve_thickness, height=full_pitch+2*sleeve_thickness) & ~assembly_region & +bottom & -top
assembly_sleeve_66.fill = zirconium

assembly_sleeve_100 = openmc.Cell(name='full assembly sleeve')
assembly_sleeve_100.region = -openmc.model.RectangularPrism(width=full_pitch+2*sleeve_thickness, height=full_pitch+2*sleeve_thickness) & ~assembly_region & +bottom & -top
assembly_sleeve_100.fill = zirconium

assembly_outer_water = openmc.Cell(name='outer water')
assembly_outer_water.region = ~assembly_sleeve_00.region & ~assembly_region & -openmc.model.RectangularPrism(width=full_pitch+2*sleeve_thickness+1, height=full_pitch+2*sleeve_thickness+1, boundary_type='reflective')
assembly_outer_water.fill = water


asm_0 = openmc.Universe(cells=[fuel_assembly_cell_0, assembly_sleeve_00, assembly_outer_water])
asm_33 = openmc.Universe(cells=[fuel_assembly_cell_33, assembly_sleeve_33, assembly_outer_water])
asm_66 = openmc.Universe(cells=[fuel_assembly_cell_66, assembly_sleeve_66, assembly_outer_water])
asm_100 = openmc.Universe(cells=[fuel_assembly_cell_100, assembly_sleeve_100, assembly_outer_water])

core_lattice = openmc.RectLattice(name='3x3 Core Lattice')
core_lattice.pitch = (full_pitch + 2*sleeve_thickness, full_pitch + 2*sleeve_thickness)
core_lattice.lower_left = (-3/2*core_lattice.pitch[0], -3/2*core_lattice.pitch[1])

core_lattice.universes = [
                         [asm_66, asm_0, asm_66],
                         [asm_0, asm_100, asm_0],
                         [asm_66, asm_0, asm_66]
]

core_width = 3*core_lattice.pitch[0]
core_height = 3*core_lattice.pitch[1]
core_region = -openmc.model.RectangularPrism(width=core_width, height=core_height, boundary_type='reflective') & +bottom & -top

core_cell = openmc.Cell(name='core', fill=core_lattice, region=core_region)

root_universe = openmc.Universe(name='root universe', cells=[core_cell])

geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()

##############################################
                # Settings #
##############################################

source_x_halfwidth = (full_pitch + 2*sleeve_thickness) * 3/2
source_y_halfwidth = (full_pitch + 2*sleeve_thickness) * 3/2

bounds = [-source_x_halfwidth, -source_y_halfwidth, -1, source_x_halfwidth, source_y_halfwidth, 1]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)

settings = openmc.Settings()
settings.source = openmc.Source(space=uniform_dist)
settings.batches = 100
settings.inactive = 10
settings.particles = 10000
settings.run_mode = 'eigenvalue'
settings.export_to_xml()

##############################################
                # Tallies #
##############################################

num_pins = 17 * 3
core_size = pitch * num_pins

mesh = openmc.RegularMesh()
mesh.dimension = [num_pins, num_pins, 1]
mesh.lower_left = [-core_size/2, -core_size/2, -1]
mesh.upper_right = [core_size/2, core_size/2, 1]

cell_filter = openmc.CellFilter(fuel_cell)

fuel_tally = openmc.Tally(name='fuel_reactions')
fuel_tally.filters = [cell_filter]
fuel_tally.nuclides = ['U235']
fuel_tally.scores = ['total', 'fission', 'absorption', '(n,gamma)']

flux_tally = openmc.Tally(name='flux')
flux_tally.filters = [openmc.MeshFilter(mesh)]
flux_tally.scores = ['flux']

rr_tally = openmc.Tally(name='reaction_rates')
rr_tally.scores = ['fission', 'absorption']

power_tally = openmc.Tally(name='radial_power')
power_tally.filters = [openmc.MeshFilter(mesh)]
power_tally.nuclides = ['U235']
power_tally.scores = ['fission']

tallies = openmc.Tallies([fuel_tally, flux_tally, rr_tally, power_tally])
tallies.export_to_xml()

##############################################
                # Plotting #
##############################################

########### Geometry Vizualization ##########

plot = openmc.Plot()
plot.filename = 'core_xy'        # Output filename will be 'core_xz.ppm'
plot.origin = (0.0, 0.0, 60.0)   # z=60 cm is typically in the middle of the fuel/control rod
plot.width = (300.0, 300.0)      # Width in x and z directions (adjust as needed)
plot.pixels = (1920, 1920)         # Resolution of the image
plot.basis = 'xy'                # Slice in the x-z plane
plot.filename = 'fuel_assembly_plot'

plots = openmc.Plots([plot])
plots.export_to_xml()
openmc.plot_geometry()


##############################################
                # Run #
##############################################

openmc.run()

##############################################
                # Post-Processing #
##############################################

import openmc
import matplotlib.pyplot as plt

sp = openmc.StatePoint('statepoint.100.h5')

power_tally = sp.get_tally(name='radial_power')
dfr = power_tally.get_pandas_dataframe()
power = dfr['mean'].values.reshape((num_pins, num_pins))

plt.figure(figsize=(6,6))
plt.imshow(power, origin='lower', cmap='hot', extent=[-core_size/2, core_size/2, -core_size/2, core_size/2])
plt.colorbar(label='Fission Rate')
plt.title('Radial Power Distribution (Fission Rate)')
plt.xlabel("x [cm]")
plt.ylabel("y [cm]")
plt.savefig("radial_power_profile.png", dpi=300)

flux_tally = sp.get_tally(name='flux')

df = flux_tally.get_pandas_dataframe()
flux = df['mean'].values.reshape((num_pins, num_pins))

plt.figure(figsize=(6,6))
plt.imshow(flux, origin='lower')
plt.colorbar(label='Flux')
plt.title('Flux Distribution')
plt.savefig("full_core_flux_plot.png")


