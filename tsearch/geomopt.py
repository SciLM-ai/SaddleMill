import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import zipfile
from ase.io import Trajectory
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from tsearch.tools import parse_inputfile, load_calculator, load_optimizer


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)
Optimizer = load_optimizer(config_dict)


def relax_structure(config_dict, atoms, logfile, trajfile):
    opt = Optimizer(FrechetCellFilter(atoms), logfile=logfile, trajectory=trajfile,
                    **config_dict[config_dict["Main"]["Optimizer"]])
    converged = opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
    return converged


def geomopt(i, config_dict, atoms, executorlib_worker_id=None):
    
    rank = executorlib_worker_id
    atoms.calc = calc

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{status_msg}\n")

    # --- MAIN LOOP ---
    with Trajectory(my_output_file, 'a') as writer:

        temp_opt_log = f'optimization_r{rank}_{i}.log'
        temp_traj = f'optimization_r{rank}_{i}.traj'
        temp_files = [temp_opt_log, temp_traj]

        try:
            converged = relax_structure(config_dict, atoms, temp_opt_log, temp_traj)
            
            if converged:
                log_status("converged")
                atoms.info['converged'] = 1
            else:
                log_status("not_converged")
                atoms.info['converged'] = 0
            atoms.info['src_index'] = i

            writer.write(atoms)

            # Clean up temp files
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                #if not converged:
                if True:  # converged or unconverged
                    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for f_name in existing_files:
                            zf.write(f_name, arcname=f"{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)

        except Exception as e:
            print(f"Rank {rank} FAILED on structure {i}: {e}")
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)
            log_status("error") 


def doublegeomopt(i, config_dict, atoms, executorlib_worker_id=None):

    rank = executorlib_worker_id
    #atoms.calc = calc

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(image_num, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{image_num},{status_msg}\n")


    with Trajectory(my_output_file, 'a') as writer:
        temp_files = []
        try:
            if not atoms.info['converged']:
                raise Exception("The given atoms object was unconverged. Skipping...")
            parent_source_idx = atoms.info['src_index']
            refined_eigenmode = atoms.info['eigenmode']

            # Save the refined TS structure data (update info but keep position)
            ts_atoms = atoms.copy()
            ts_atoms.info = atoms.info.copy()
            ts_atoms.calc = SinglePointCalculator(
                ts_atoms, 
                energy=atoms.get_potential_energy(), 
                forces=atoms.get_forces()
            )

            # MINIMIZATION 1 (Forward along mode)
            temp_opt_log = f'optimization_r{rank}_{i}-0.log'
            temp_traj = f'optimization_r{rank}_{i}-0.traj'
            temp_files = [temp_opt_log, temp_traj]
                            
            min1 = ts_atoms.copy()
            min1.calc = calc
            displacement = 0.25 # Angstrom (Small push)
            min1.positions += displacement * refined_eigenmode
            conv1 = relax_structure(config_dict, min1, temp_opt_log, temp_traj)
            e, f = min1.get_potential_energy(), min1.get_forces()
            min1.calc = SinglePointCalculator(
                min1,
                energy=e,
                forces=f
            )
            min1.info['type'] = 'minimum_1'
            min1.info['parent_ts_index'] = parent_source_idx
            min1.info['converged'] = conv1

            # MINIMIZATION 2 (Backward along mode)
            temp_opt_log = f'optimization_r{rank}_{i}-1.log'
            temp_traj = f'optimization_r{rank}_{i}-1.traj'
            temp_files.append(temp_opt_log)
            temp_files.append(temp_traj)

            min2 = ts_atoms.copy()
            min2.calc = calc
            min2.positions -= displacement * refined_eigenmode
            conv2 = relax_structure(config_dict, min2, temp_opt_log, temp_traj)
            e, f = min2.get_potential_energy(), min2.get_forces()
            min2.calc = SinglePointCalculator(
                min2,
                energy=e,
                forces=f
            )
            min2.info['type'] = 'minimum_2'
            min2.info['parent_ts_index'] = parent_source_idx
            min2.info['converged'] = conv2

            # 4. WRITE TRIPLET (Min1 -> TS -> Min2)
            writer.write(min1)
            writer.write(ts_atoms)
            writer.write(min2)
            
            print(f"\tRank {rank}: File {filename} Img {idx} -> Done ({idx+1}/{len(traj_images)}).", flush=True)

                        except Exception as e:
                            print(f"Rank {rank}: Failed on {filename} image {idx}. Error: {e}", flush=True)
                            continue

    print(f"Rank {rank}: Finished processing assigned files.")
