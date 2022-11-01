import os
import land_and_sea_functions as lnz

# this does not get set for os.system invoked code
# lnz.set_terrain_type_goal( [0.7,0.3] )
# lnz.set_terrain_surface_goal( 0.2 )

os.system('python lnz_rand_via_runner.py v1')
os.system('python lnz_rand_via_runner.py v2')
os.system('python lnz_rand_via_runner.py v3')
os.system('python lnz_rand_via_runner.py v4')
os.system('python lnz_rand_via_runner.py v5')

os.system('python lnz_pics_via_runner.py v1')
os.system('python lnz_pics_via_runner.py v2')
os.system('python lnz_pics_via_runner.py v3')
os.system('python lnz_pics_via_runner.py v4')
os.system('python lnz_pics_via_runner.py v5')

os.system('python lnz_double_output_via_runner.py')
