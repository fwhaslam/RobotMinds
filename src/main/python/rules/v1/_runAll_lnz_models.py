import os

def set_environment():
    os.putenv( 'TERRAIN_TYPE_GOAL', '[0.7,0.3]' )
    os.putenv( 'TERRAIN_SURFACE_GOAL', '0.2' )
    return

def run_all():

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
    return

if __name__ == "__main__":
    set_environment()
    run_all()
