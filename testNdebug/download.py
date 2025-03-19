import boto3
import botocore
import os
from pathlib import Path
import csv


def download():
    OUTPUT_BASE = Path("data/downloaded")

    with open("data/1200_ids.csv", "r") as f:
        hcp1200_ids = [line[0] for line in csv.reader(f)]

    with open("data/7T_ids.csv", "r") as f:
        hcp7T_ids = [line[0] for line in csv.reader(f)]

    hcp_intersect = [subj for subj in hcp1200_ids if subj in hcp7T_ids]

    subject_list = [hcp_intersect[0]]

    rest_runs = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

    surface_types = ["inflated", "sphere", "roi", "midthickness", "white", "aparc"]
    file_types = ["surf", "func", "label"]
    hemispheres = ["L", "R"]

    session = boto3.Session(profile_name="hcp")
    s3 = session.client("s3")
    bucket_name = "hcp-openaccess"

    for subj in subject_list:
        print(f"Processing subject: {subj}")
        
        subj_path = OUTPUT_BASE / subj / "hcp1200"
        try :
            fsaverage_path = f"{subj_path}/fsaverage_LR32k/"
            Path(fsaverage_path).mkdir(parents=True, exist_ok=True)

            # List of `.gii` files to download
            gii_files = [
                
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.midthickness.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.midthickness.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.white.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.white.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.inflated.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.inflated.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.sphere.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.sphere.32k_fs_LR.surf.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.BA.32k_fs_LR.label.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.BA.32k_fs_LR.label.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.L.aparc.32k_fs_LR.label.gii",
                f"HCP_1200/{subj}/MNINonLinear/fsaverage_LR32k/{subj}.R.aparc.32k_fs_LR.label.gii"
            ]

            # Download each `.gii` file
            for gii_file in gii_files:
                file_name = gii_file.split("/")[-1]  # Extract file name
                if not file_name:  # If file_name is empty, skip this entry.
                    continue
                local_path = fsaverage_path + file_name
                print(f"Downloading: s3://{bucket_name}/{gii_file} -> {local_path}")
                with open(local_path, "wb") as f:
                    s3.download_fileobj("hcp-openaccess", gii_file, f)
                    

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                continue
            else:
                raise e

        for run_name in rest_runs:
            run_prefix = f"HCP_1200/{subj}/MNINonLinear/Results/{run_name}/"
            local_run_dir = OUTPUT_BASE / subj / "hcp1200" / "rest" / run_name
            local_run_dir.mkdir(parents=True, exist_ok=True)

            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=run_prefix)

            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".dtseries.nii") or key.endswith(".surf.gii"):
                        filename = os.path.basename(key)
                        local_path = local_run_dir / filename
                        print(f"Downloading: {key} -> {local_path}")
                        try:
                            s3.download_file(bucket_name, key, str(local_path))
                        except botocore.exceptions.ClientError as e:
                            print(f"Failed: {key}, Reason: {e}")


        label_dir = OUTPUT_BASE / subj / "hcp1200" / "T1w" / "label"
        label_dir.mkdir(parents=True, exist_ok=True)

        label_files = [
            f"HCP_1200/{subj}/T1w/{subj}/label/lh.cortex.label",
            f"HCP_1200/{subj}/T1w/{subj}/label/lh.aparc.annot",
            f"HCP_1200/{subj}/T1w/{subj}/label/rh.cortex.label",
            f"HCP_1200/{subj}/T1w/{subj}/label/rh.aparc.annot",
        ]


    import subprocess
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dtseries_path = os.path.join(current_dir, "data", "downloaded", "100610", "hcp1200", "rest", "rfMRI_REST1_LR", "rfMRI_REST1_LR_Atlas.dtseries.nii")
    out_L_path = os.path.join(current_dir, "data", "downloaded", "100610", "hcp1200", "rest", "rfMRI_REST1_LR", "rfMRI_REST1_LR_Atlas.L.func.gii")
    out_R_path = os.path.join(current_dir, "data", "downloaded", "100610", "hcp1200", "rest", "rfMRI_REST1_LR", "rfMRI_REST1_LR_Atlas.R.func.gii")

    command = [
        "wb_command",
        "-cifti-separate",
        dtseries_path,
        "COLUMN",
        "-metric",
        "CORTEX_LEFT",
        out_L_path,
        "-metric",
        "CORTEX_RIGHT",
        out_R_path
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Command ran successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")