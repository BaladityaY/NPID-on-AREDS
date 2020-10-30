'''
Script: symlink_dir_maker
Purpose: Create a softlink directory tree with certain classes excluded.
Author: Aaron Reite
'''

import os

exclude_list = \
    ["Building",
     "Building_AircraftHangar",
     "Building_Damaged",
     "Building_Facility",
     "Building_Hut-Tent",
     "Building_Shed",
     "ConstructionSite",
     "ContainerLot",
     "VehicleLot",
     "RV", # excluded because only 17 in train & 2 in val
     "PV", # exluded because it's almost identical to small car
     "TowerStructure", # also a building
]

exclude_list = []

# Script Variables
source_chip_dir = '/raid/etegent/xview/xview_chips/goley/ufl_square'
dest_chip_dir = '/raid/etegent/xview/xview_chips/areite/ufl_square_all'

if __name__ == '__main__':

    for split_dir in ('train', 'val'):

        source = os.path.join(source_chip_dir, split_dir)
        dest = os.path.join(dest_chip_dir, split_dir)

        print("Writing symlinks from", source)
        print("to", dest)

        # Create train or val dir
        if not os.path.isdir(dest):
            os.mkdir(dest)

        total_files = 0
        
        dirs = os.listdir(source)
        dirs.sort()
        for d in dirs:

            if d in exclude_list:
                continue

            source_dir_path = os.path.join(source, d)
            dest_dir_path = os.path.join(dest, d)

            # Create class dir
            os.mkdir(dest_dir_path)

            file_count = 0
            for f in os.listdir(source_dir_path):

                source_file_path = os.path.join(source_dir_path, f)
                dest_file_path = os.path.join(dest_dir_path, f)

                os.symlink(source_file_path, dest_file_path)
                file_count += 1

            total_files += file_count
            print(d + ": " + str(file_count))
        print("TOTAL:", total_files)
