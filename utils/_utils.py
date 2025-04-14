import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk

def view_image_stack(path):
    img = nib.load(path)
    img_f = img.get_fdata()
    fig, axes = plt.subplots(img_f.shape[2],1, figsize=(100,100))
    for ind, i in enumerate(range(img_f.shape[2])):
        axes[ind].imshow(img_f[:,:,i], cmap="bone")
    plt.show()
    
def show_sitk_image(image, slice_index=0):
    # Convert SimpleITK image to a NumPy array for visualization
    img_array = sitk.GetArrayFromImage(image)
    
    # Check if the image is 3D (multiple slices)
    if len(img_array.shape) == 3:
        # Show the specified slice (default is slice 0)
        plt.imshow(img_array[slice_index, :, :], cmap='gray')
    else:
        # If it's a 2D image, just display it
        plt.imshow(img_array, cmap='gray')
    
    plt.axis('off')  # Turn off axis labels
    plt.show()