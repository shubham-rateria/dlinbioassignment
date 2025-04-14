# MR Background

1. **ADC** (Apparent Diffusion Coefficient):  
   - A type of diffusion-weighted imaging (DWI) that measures the magnitude of water diffusion within tissues. ADC maps are helpful in identifying cancerous tissues, as they tend to restrict water movement.

2. **COR** (Coronal):  
   - Refers to the coronal plane or orientation of the image. In this orientation, the MRI slices are taken vertically from the front (anterior) to the back (posterior) of the body.

3. **HBV** (High b-value DWI):  
   - Refers to diffusion-weighted imaging with a high "b-value," which controls the sensitivity to diffusion. High b-value images are particularly useful in highlighting areas with restricted diffusion, such as tumors.

4. **SAG** (Sagittal):  
   - Refers to the sagittal plane or orientation, where MRI slices are taken vertically from the side of the body, dividing it into left and right halves.

5. **T2W** (T2-Weighted):  
   - A type of MRI sequence that emphasizes the differences in the T2 relaxation times of tissues. T2-weighted images are useful for identifying edema, fluid-filled structures, and some tumors.

These modalities provide a comprehensive view of tissues, helping in the diagnosis and staging of diseases like cancer.

# Dataset Information

## Dataset Overview

| Dataset       | Description                                              | Number of Cases |
|---------------|----------------------------------------------------------|-----------------|
| PICAI         | Prostate Imaging Cancer AI dataset                       | 1295            |
| WCM           | Weill Cornell Medicine diagnostic dataset                | 524             |
| Prostate-MRI  | Prostate MRI-US Biopsy dataset                           | 605             |



## Directory Structure and Details

### PICAI
- **Path:** `/share/sablab/nfs04/data/PICAI`
- **Structure:**
  - `nifti`: Where images are saved
    - `convertedADC`: ADC images
    - `convertedDWI`: DWI images
    - `convertedT2`: T2 images
  - Segmentation labels: `/share/sablab/nfs04/data/PICAI/picai_labels/csPCa_lesion_delineations/human_expert`
  - Demo file: `/share/sablab/nfs04/users/pa369/demo-picai-2.csv`
    - **Notes:**
      - Filenames are determined by the `ptid_sid` column.
      - `case_csPCa`: Cancer label column (e.g., YES for cancer).
      - `trainvaltest`: Train/validation/test splits.
      - Columns `t2`, `dwi`, `adc` include filenames for each modality.
      - `human_expert` shows images with expert-annotated segmentation labels (1295 out of 1500). 
      - Empty label images may exist for healthy patients. Please verify.

### Prostate-MRI-US-Biopsy
- **Path:** `/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy`
- **Structure:**
  - `nifti`:
    - `adc`: ADC images
    - `t2`: T2 images
    - `dwi`: DWI images
  - Demo file: `/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv`
    - **Notes:**
      - Filenames are determined by `pid` (e.g., `[pid]_1000000_*.nii.gz`).
      - `cancer`: Cancer label column (e.g., True for cancer).
      - `trainvaltest`: Train/validation/test splits.
      - Columns `t2`, `dwi`, `adc` include filenames for each modality.
      - Every cancer-positive patient has segmentation labels: `/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR`.

### WCM
- **Path:** `/share/sablab/nfs04/data/prostate-wcm-diagnostic`
- **Structure:**
  - `nifti`:
    - `adc`: ADC images
    - `t2`: T2 images
    - `dwi`: DWI images
  - Demo file: `/share/sablab/nfs04/data/prostate-wcm-diagnostic/demo/demo-wcm-classification-bval1000orhigher-single-target.csv`
    - **Notes:**
      - `pid`: Patient ID.
      - `cspca`: Cancer label column (e.g., True for cancer).
      - `trainvaltest`: Train/validation/test splits.
      - Columns `t2name`, `dwiname`, `adcname` include filenames for each modality.
      - `maskname` includes filenames for segmentation masks (cancer lesion). Only use for cancer-positive patients.


## nnUNet Gland Segmentation Results

| Dataset       | Directory Path                                                                                                                |
|---------------|-------------------------------------------------------------------------------------------------------------------------------|
| PICAI         | `/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset002_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4` |
| WCM           | `/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset006_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4` |
| Biopsy        | `/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset004_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4` |

Linking dataset to segmentation:

PICAI
| Patient ID | File Name                       | Segmentation File               |
|------------|---------------------------------|----------------------------------|
| 10533      | 10533_1000543_0000.nii.gz      | 10533_1000543.nii.gz            |

WCM
| File Name           | Segmentation File               | Meta Data (PSA, Vol) |
|---------------------|----------------------------------| -------------- |
| WC-PIRADS-0015      | 90015_0000000.nii.gz            | demo-wcm-classification-bval1000orhigher-single-target |

Prostate-MRI
| Patient ID (pid) | File Name                       | Segmentation File               | Meta Data (PSA, Vol) |
|------------|---------------------------------|----------------------------------|-----------------------------------|
| 80002      | 80002_1000000_*.nii.gz          | 80002_1000000.nii.gz            | demo-tcia-mr-us-matching.csv |

---

## Dataset Details

| Key | Description |
| --- | --- |
| dataset | The name of the dataset. |
| base_file_path | The base path where the files are located. |
| seg_path_t2 | The path to the T2 segmentation file. |
| seg_path_t2_ai | The path to the T2 segmentation file generated by AI. |
| seg_path_dwi | The path to the DWI segmentation file. |
| seg_path_adc | The path to the ADC segmentation file. |
| t2_filename | The filename of the T2 image. |
| dwi_filename | The filename of the DWI image. |
| adc_filename | The filename of the ADC image. |
| cancer | Indicates if the patient has cancer. |
| seg_slice_start | The starting slice of the segmentation. |
| seg_slice_end | The ending slice of the segmentation. |
| max_area_slice | The slice with the maximum area. |
| pseg_path | The path to the prostate gland segmentation file. |
| pseg_max_area_slice | The slice with the maximum area of the prostate gland segmentation. |
| pseg_slice_start | The starting slice of the prostate gland segmentation. |
| pseg_slice_end | The ending slice of the prostate gland segmentation. |
| dataset_gland_vol | The volume of the prostate gland provided by the dataset. |
| gland_vol | The volume of the prostate gland calculated in pixel space. Currently this is calculated as a percentage of the total prostate in pixel space. |
| psa | The PSA (Prostate-Specific Antigen) level in ng/mL. |
| psad | The PSA density. |

---

Open Questions:
1. How do we validate if the model has correctly generated required PSA and volume? Can we establish a correlation between dataset volume and pixel based volume?