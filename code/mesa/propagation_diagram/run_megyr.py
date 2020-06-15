import megyr
import os

def run_megyr(masses,outdir):
    megyr.run({
        "input": {
            "mesa_configs": ["inlist.mustache"]
        },
        "output": {
            "mesa_profile_summary_file": "mesa_profile_attributes.csv",
            "output_dir" : outdir
        },
        "stages": {
            "mesa_params": {
                "initial_mass": masses
            }
        }
    })
    
if __name__ == '__main__':
    outdir1 = 'out'
    masses = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    #run_megyr(masses,outdir1) #already have PMS models
    os.rename("inlist.mustache","inlist_pre.mustache")
    os.rename("inlist_post.mustache","inlist.mustache")
    #os.remove("completed_tasks.csv")
    outdir2 = 'out_post'
    try:
        os.mkdir(outdir2)
    except FileExistsError:
        pass
    for m in masses:
        try:
            os.mkdir(outdir2+f'/mesa_initial_mass_{m}__')
        except FileExistsError:
            pass
        os.system(f'cp {outdir1}/mesa_initial_mass_{m}__/{m}M_at_ZAMS.mod {outdir2}/mesa_initial_mass_{m}__/{m}M_at_ZAMS.mod')
        os.system(f'cp profile_columns.list {outdir2}/mesa_initial_mass_{m}__/profile_columns.list')
    run_megyr(masses,outdir2)
    os.rename("inlist.mustache","inlist_post.mustache")
    os.rename("inlist_pre.mustache","inlist.mustache")