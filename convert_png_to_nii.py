import nibabel as nib

picai_1 = "/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset002_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/10715_1000731.nii.gz"
picai_2 = "/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset002_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/10694_1000710.nii.gz"
picai_3 = "/share/sablab/nfs04/users/pa369/winning_paper_nnunet/nnUNet_results/Dataset002_gland_segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/10482_1000490.nii.gz"

# original_nifti_path = "original.nii.gz"  # Path to the original NIfTI file
original_nifti = nib.load(picai_3)

# Get the original affine matrix
original_affine = original_nifti.affine
print("Original Affine Matrix:\n", original_affine)
