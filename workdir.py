# This is only a utils file to change the working directory to the root directory of the project.
# And import some user-defined modules.
# All utils files in the playground directory should be the same.
import os
import sys


def get_working_dir(current_dir, endswith):
    if current_dir.endswith(endswith):
        return current_dir
    temp_dir_name = os.path.dirname(current_dir)
    for _ in range(100):
        if temp_dir_name.endswith(endswith):
            break
        temp_dir_name = os.path.dirname(temp_dir_name)
    return temp_dir_name


def change_workdir(endswith, print_cwd=False):
    file_path = os.getcwd()
    # repo root
    working_dir = get_working_dir(file_path, endswith)
    os.chdir(working_dir)
    sys.path.append(working_dir)
    if print_cwd:
        print(f"Current work dir: {working_dir}")
    return working_dir
