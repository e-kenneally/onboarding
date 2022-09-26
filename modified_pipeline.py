%YAML 1.1
---
# CPAC Pipeline Configuration YAML file
# Version 1.8.4
#
# http://fcp-indi.github.io for more info.
#
# Tip: This file can be edited manually with a text editor for quick modifications.

pipeline_setup:

  # Name for this pipeline configuration - useful for identification.
  pipeline_name: cpac-default-pipeline

  output_directory:

    # Directory where C-PAC should write out processed data, logs, and crash reports.
    # - If running in a container (Singularity/Docker), you can simply set this to an arbitrary
    #   name like '/outputs', and then map (-B/-v) your desired output directory to that label.
    # - If running outside a container, this should be a full path to a directory.
    path: /outputs/output

    # (Optional) Path to a BIDS-Derivatives directory that already has outputs.
    #   - This option is intended to ingress already-existing resources from an output
    #     directory without writing new outputs back into the same directory.
    #   - If provided, C-PAC will ingress the already-computed outputs from this directory and
    #     continue the pipeline from where they leave off.
    #   - If left as 'None', C-PAC will ingress any already-computed outputs from the
    #     output directory you provide above in 'path' instead, the default behavior.
    source_outputs_dir: None

    # Set to True to make C-PAC ingress the outputs from the primary output directory if they
    # exist, even if a source_outputs_dir is provided
    #   - Setting to False will pull from source_outputs_dir every time, over-writing any
    #     calculated outputs in the main output directory
    #   - C-PAC will still pull from source_outputs_dir if the main output directory is
    #     empty, however
    pull_source_once: True

    # Include extra versions and intermediate steps of functional preprocessing in the output directory.
    write_func_outputs: False

    # Include extra outputs in the output directory that may be of interest when more information is needed.
    write_debugging_outputs: False

    # Output directory format and structure.
    # Options: default, ndmg
    output_tree: "default"

    # Quality control outputs
    quality_control:
      # Generate quality control pages containing preprocessing and derivative outputs.
      generate_quality_control_images: True

      # Generate eXtensible Connectivity Pipeline-style quality control files
      generate_xcpqc_files: False

  working_directory:

    # Directory where C-PAC should store temporary and intermediate files.
    # - This directory must be saved if you wish to re-run your pipeline from where you left off (if not completed).
    # - NOTE: As it stores all intermediate files, this directory can grow to become very
    #   large, especially for data with a large amount of TRs.
    # - If running in a container (Singularity/Docker), you can simply set this to an arbitrary
    #   name like '/work', and then map (-B/-v) your desired output directory to that label.
    # - If running outside a container, this should be a full path to a directory.
    # - This can be written to '/tmp' if you do not intend to save your working directory.
    path: /outputs/working

    # Deletes the contents of the Working Directory after running.
    # This saves disk space, but any additional preprocessing or analysis will have to be completely re-run.
    remove_working_dir: True

  log_directory:

    # Whether to write log details of the pipeline run to the logging files.
    run_logging: True

    path: /outputs/logs

  crash_log_directory:

    # Directory where CPAC should write crash logs.
    path: /outputs/crash

  system_config:

    # Random seed used to fix the state of execution.
    # If unset, each process uses its own default.
    # If set, a `random.log` file will be generated logging the random seed and each node to which that seed was applied.
    # If set to a positive integer (up to 2147483647), that integer will be used to seed each process that accepts a random seed.
    # If set to 'random', a random positive integer (up to 2147483647) will be generated and that seed will be used to seed each process that accepts a random seed.
    random_seed:

    # Select Off if you intend to run CPAC on a single machine.
    # If set to On, CPAC will attempt to submit jobs through the job scheduler / resource manager selected below.
    on_grid:

      run: Off

      # Sun Grid Engine (SGE), Portable Batch System (PBS), or Simple Linux Utility for Resource Management (SLURM).
      # Only applies if you are running on a grid or compute cluster.
      resource_manager: SGE

      SGE:
        # SGE Parallel Environment to use when running CPAC.
        # Only applies when you are running on a grid or compute cluster using SGE.
        parallel_environment:  mpi_smp

        # SGE Queue to use when running CPAC.
        # Only applies when you are running on a grid or compute cluster using SGE.
        queue:  all.q

    # The maximum amount of memory each participant's workflow can allocate.
    # Use this to place an upper bound of memory usage.
    # - Warning: 'Memory Per Participant' multiplied by 'Number of Participants to Run Simultaneously'
    #   must not be more than the total amount of RAM.
    # - Conversely, using too little RAM can impede the speed of a pipeline run.
    # - It is recommended that you set this to a value that when multiplied by
    #   'Number of Participants to Run Simultaneously' is as much RAM you can safely allocate.
    maximum_memory_per_participant: 1

    # Prior to running a pipeline C-PAC makes a rough estimate of a worst-case-scenario maximum concurrent memory usage with high-resoltion data, raising an exception describing the recommended minimum memory allocation for the given configuration.
    # Turning this option off will allow pipelines to run without allocating the recommended minimum, allowing for more efficient runs at the risk of out-of-memory crashes (use at your own risk)
    raise_insufficient: On

    # A callback.log file from a previous run can be provided to estimate memory usage based on that run.
    observed_usage:
      # Path to callback log file with previously observed usage.
      # Can be overridden with the commandline flag `--runtime_usage`.
      callback_log:
      # Percent. E.g., `buffer: 10` would estimate 1.1 * the observed memory usage from the callback log provided in "usage".
      # Can be overridden with the commandline flag `--runtime_buffer`.
      buffer: 10

    # The maximum amount of cores (on a single machine) or slots on a node (on a cluster/grid)
    # to allocate per participant.
    # - Setting this above 1 will parallelize each participant's workflow where possible.
    #   If you wish to dedicate multiple cores to ANTS-based anatomical registration (below),
    #   this value must be equal or higher than the amount of cores provided to ANTS.
    # - The maximum number of cores your run can possibly employ will be this setting multiplied
    #   by the number of participants set to run in parallel (the 'Number of Participants to Run
    #   Simultaneously' setting).
    max_cores_per_participant: 1

    # The number of cores to allocate to ANTS-based anatomical registration per participant.
    # - Multiple cores can greatly speed up this preprocessing step.
    # - This number cannot be greater than the number of cores per participant.
    num_ants_threads: 1

    # The number of cores to allocate to processes that use OpenMP.
    num_OMP_threads: 1

    # The number of participant workflows to run at the same time.
    # - The maximum number of cores your run can possibly employ will be this setting
    #   multiplied by the number of cores dedicated to each participant (the 'Maximum Number of Cores Per Participant' setting).
    num_participants_at_once: 1

    # Full path to the FSL version to be used by CPAC.
    # If you have specified an FSL path in your .bashrc file, this path will be set automatically.
    FSLDIR:  /usr/share/fsl/5.0

  Amazon-AWS:

    # If setting the 'Output Directory' to an S3 bucket, insert the path to your AWS credentials file here.
    aws_output_bucket_credentials:

    # Enable server-side 256-AES encryption on data to the S3 bucket
    s3_encryption: False

  Debugging:

    # Verbose developer messages.
    verbose: Off


# PREPROCESSING
# -------------
surface_analysis:

  # Will run Freesurfer for surface-based analysis. Will output traditional Freesurfer derivatives.
  # If you wish to employ Freesurfer outputs for brain masking or tissue segmentation in the voxel-based pipeline,
  # select those 'Freesurfer-' labeled options further below in anatomical_preproc.
  freesurfer: 

    # If anatomical_preproc['brain_extraction']['using'] includes FreeSurfer-ABCD and this switch is On, C-PAC will automatically turn this switch Off to avoid running FreeSurfer twice unnecessarily
    run: Off

    # Add extra arguments to recon-all command
    reconall_args: None
    
    # (Optional) Provide an already-existing FreeSurfer output directory to ingress already-computed surfaces
    freesurfer_dir: None

  # Run ABCD-HCP post FreeSurfer and fMRISurface pipeline
  post_freesurfer: 

    run: Off

    subcortical_gray_labels: /opt/dcan-tools/pipeline/global/config/FreeSurferSubcorticalLabelTableLut.txt

    freesurfer_labels: /opt/dcan-tools/pipeline/global/config/FreeSurferAllLut.txt

    surf_atlas_dir: /opt/dcan-tools/pipeline/global/templates/standard_mesh_atlases

    gray_ordinates_dir: /opt/dcan-tools/pipeline/global/templates/91282_Greyordinates

    gray_ordinates_res: 2

    high_res_mesh: 164

    low_res_mesh: 32

    fmri_res: 2

    smooth_fwhm: 2


longitudinal_template_generation:

  # If you have multiple T1w's, you can generate your own run-specific custom
  # T1w template to serve as an intermediate to the standard template for
  # anatomical registration.

  # This runs before the main pipeline as it requires multiple T1w sessions
  # at once.
  run: Off

  # Freesurfer longitudinal template algorithm using FSL FLIRT
  # Method to average the dataset at each iteration of the template creation
  # Options: median, mean or std
  average_method: median

  # Degree of freedom for FLIRT in the template creation
  # Options: 12 (affine), 9 (traditional), 7 (global rescale) or 6 (rigid body)
  dof: 12

  # Interpolation parameter for FLIRT in the template creation
  # Options: trilinear, nearestneighbour, sinc or spline
  interp: trilinear

  # Cost function for FLIRT in the template creation
  # Options: corratio, mutualinfo, normmi, normcorr, leastsq, labeldiff or bbr
  cost: corratio

  # Number of threads used for one run of the template generation algorithm
  thread_pool: 2

  # Threshold of transformation distance to consider that the loop converged
  # (-1 means numpy.finfo(np.float64).eps and is the default)
  convergence_threshold: -1


anatomical_preproc:

  run: On

  run_t2: Off

  # Non-local means filtering via ANTs DenoiseImage
  non_local_means_filtering: 

    # this is a fork option
    run: [Off]

    # options: 'Gaussian' or 'Rician'
    noise_model: 'Gaussian'

  # N4 bias field correction via ANTs
  n4_bias_field_correction:

    # this is a fork option
    run: [Off]

    # An integer to resample the input image to save computation time. Shrink factors <= 4 are commonly used.
    shrink_factor: 2

  # Bias field correction based on square root of T1w * T2w
  t1t2_bias_field_correction: 

    run: Off 
    
    BiasFieldSmoothingSigma: 5
    
  acpc_alignment:

    run: Off

    # Run ACPC alignment before non-local means filtering or N4 bias
    # correction
    run_before_preproc: True

    # ACPC size of brain in z-dimension in mm.
    # Default: 150mm for human data.
    brain_size: 150

    # Choose a tool to crop the FOV in ACPC alignment. 
    # Using FSL's robustfov or flirt command. 
    # Default: robustfov for human data, flirt for monkey data. 
    FOV_crop: robustfov
    
    # ACPC Target
    # options: 'brain' or 'whole-head'
    #   note: 'brain' requires T1w_brain_ACPC_template below to be populated
    acpc_target: 'whole-head'

    # Run ACPC alignment on brain mask 
    # If the brain mask is in native space, turn it on
    # If the brain mask is ACPC aligned, turn it off
    align_brain_mask: Off

    # ACPC aligned template
    T1w_ACPC_template: /usr/share/fsl/5.0/data/standard/MNI152_T1_1mm.nii.gz
    T1w_brain_ACPC_template: /usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz
    T2w_ACPC_template: None
    T2w_brain_ACPC_template: None

  brain_extraction:
  
    run: On

    # using: ['3dSkullStrip', 'BET', 'UNet', 'niworkflows-ants', 'FreeSurfer-ABCD', 'FreeSurfer-BET-Tight', 'FreeSurfer-BET-Loose']
    # this is a fork option
    using: ['3dSkullStrip']

    # option parameters
    AFNI-3dSkullStrip:

      # Output a mask volume instead of a skull-stripped volume. The mask volume containes 0 to 6, which represents voxel's postion. If set to True, C-PAC will use this output to generate anatomical brain mask for further analysis.
      mask_vol: False

      # Set the threshold value controlling the brain vs non-brain voxels. Default is 0.6.
      shrink_factor: 0.6

      # Vary the shrink factor at every iteration of the algorithm. This prevents the likelihood of surface getting stuck in large pools of CSF before reaching the outer surface of the brain. Default is On.
      var_shrink_fac: True

      # The shrink factor bottom limit sets the lower threshold when varying the shrink factor. Default is 0.4, for when edge detection is used (which is On by default), otherwise the default value is 0.65.
      shrink_factor_bot_lim: 0.4

      # Avoids ventricles while skullstripping.
      avoid_vent: True

      # Set the number of iterations. Default is 250.The number of iterations should depend upon the density of your mesh.
      n_iterations: 250

      # While expanding, consider the voxels above and not only the voxels below
      pushout: True

      # Perform touchup operations at the end to include areas not covered by surface expansion.
      touchup: True

      # Give the maximum number of pixels on either side of the hole that can be filled. The default is 10 only if 'Touchup' is On - otherwise, the default is 0.
      fill_hole: 10

      # Perform nearest neighbor coordinate interpolation every few iterations. Default is 72.
      NN_smooth: 72

      # Perform final surface smoothing after all iterations. Default is 20.
      smooth_final: 20

      # Avoid eyes while skull stripping. Default is On.
      avoid_eyes: True

      # Use edge detection to reduce leakage into meninges and eyes. Default is On.
      use_edge: True

      # Speed of expansion.
      exp_frac: 0.1

      # Perform aggressive push to edge. This might cause leakage. Default is Off.
      push_to_edge: False

      # Use outer skull to limit expansion of surface into the skull in case of very strong shading artifacts. Use this only if you have leakage into the skull.
      use_skull: Off

      # Percentage of segments allowed to intersect surface. It is typically a number between 0 and 0.1, but can include negative values (which implies no testing for intersection).
      perc_int: 0

      # Number of iterations to remove intersection problems. With each iteration, the program automatically increases the amount of smoothing to get rid of intersections. Default is 4.
      max_inter_iter: 4

      # Multiply input dataset by FAC if range of values is too small.
      fac: 1

      # Blur dataset after spatial normalization. Recommended when you have lots of CSF in brain and when you have protruding gyri (finger like). If so, recommended value range is 2-4. Otherwise, leave at 0.
      blur_fwhm: 0

      # Set it as True if processing monkey data with AFNI
      monkey: False

    FSL-BET:

      # Set the threshold value controling the brain vs non-brain voxels, default is 0.5
      frac: 0.5

      # Mask created along with skull stripping. It should be `On`, if selected functionalMasking :  ['Anatomical_Refined'] and `FSL` as skull-stripping method.
      mask_boolean: On

      # Mesh created along with skull stripping
      mesh_boolean: Off

      # Create a surface outline image
      outline: Off

      # Add padding to the end of the image, improving BET.Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      padding: Off

      # Integer value of head radius
      radius: 0

      # Reduce bias and cleanup neck. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      reduce_bias: Off

      # Eyes and optic nerve cleanup. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      remove_eyes: Off

      # Robust brain center estimation. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      robust: Off

      # Create a skull image
      skull: Off

      # Gets additional skull and scalp surfaces by running bet2 and betsurf. This is mutually exclusive with reduce_bias, robust, padding, remove_eyes
      surfaces: Off

      # Apply thresholding to segmented brain image and mask
      threshold: Off

      # Vertical gradient in fractional intensity threshold (-1,1)
      vertical_gradient : 0.0

    UNet:

      # UNet model
      unet_model : s3://fcp-indi/resources/cpac/resources/Site-All-T-epoch_36.model

    niworkflows-ants:

      # Template to be used during niworkflows-ants.
      # It is not necessary to change this path unless you intend to use a non-standard template.

      # niworkflows-ants Brain extraction template
      template_path : /ants_template/oasis/T_template0.nii.gz

      # niworkflows-ants probability mask
      mask_path : /ants_template/oasis/T_template0_BrainCerebellumProbabilityMask.nii.gz

      # niworkflows-ants registration mask (can be optional)
      regmask_path : /ants_template/oasis/T_template0_BrainCerebellumRegistrationMask.nii.gz

    FreeSurfer-BET:

      # Template to be used for FreeSurfer-BET brain extraction in CCS-options pipeline
      T1w_brain_template_mask_ccs: /ccs_template/MNI152_T1_1mm_first_brain_mask.nii.gz


segmentation:

  # Automatically segment anatomical images into white matter, gray matter,
  # and CSF based on prior probability maps.
  run: On

  tissue_segmentation:

    # using: ['FSL-FAST', 'Template_Based', 'ANTs_Prior_Based', 'FreeSurfer']
    # this is a fork point
    using: ['FSL-FAST']

    # option parameters
    FSL-FAST:

      thresholding:

        # thresholding of the tissue segmentation probability maps
        # options: 'Auto', 'Custom'
        use: 'Auto'

        Custom:
          # Set the threshold value for the segmentation probability masks (CSF, White Matter, and Gray Matter)
          # The values remaining will become the binary tissue masks.
          # A good starting point is 0.95.

          # CSF (cerebrospinal fluid) threshold.
          CSF_threshold_value : 0.95

          # White matter threshold.
          WM_threshold_value : 0.95

          # Gray matter threshold.
          GM_threshold_value : 0.95

      use_priors:

        # Use template-space tissue priors to refine the binary tissue masks generated by segmentation.
        run: On

        # Full path to a directory containing binarized prior probability maps.
        # These maps are included as part of the 'Image Resource Files' package available on the Install page of the User Guide.
        # It is not necessary to change this path unless you intend to use non-standard priors.
        priors_path: $FSLDIR/data/standard/tissuepriors/2mm

        # Full path to a binarized White Matter prior probability map.
        # It is not necessary to change this path unless you intend to use non-standard priors.
        WM_path: $priors_path/avg152T1_white_bin.nii.gz

        # Full path to a binarized Gray Matter prior probability map.
        # It is not necessary to change this path unless you intend to use non-standard priors.
        GM_path: $priors_path/avg152T1_gray_bin.nii.gz

        # Full path to a binarized CSF prior probability map.
        # It is not necessary to change this path unless you intend to use non-standard priors.
        CSF_path: $priors_path/avg152T1_csf_bin.nii.gz

    Template_Based:

      # These masks should be in the same space of your registration template, e.g. if
      # you choose 'EPI Template' , below tissue masks should also be EPI template tissue masks.
      #
      # Options: ['T1_Template', 'EPI_Template']
      template_for_segmentation: ['T1_Template']

      # These masks are included as part of the 'Image Resource Files' package available
      # on the Install page of the User Guide.

      # Full path to a binarized White Matter mask.
      WHITE: $FSLDIR/data/standard/tissuepriors/2mm/avg152T1_white_bin.nii.gz

      # Full path to a binarized Gray Matter mask.
      GRAY: $FSLDIR/data/standard/tissuepriors/2mm/avg152T1_gray_bin.nii.gz

      # Full path to a binarized CSF mask.
      CSF: $FSLDIR/data/standard/tissuepriors/2mm/avg152T1_csf_bin.nii.gz

    ANTs_Prior_Based:

      # Generate white matter, gray matter, CSF masks based on antsJointLabelFusion
      # ANTs Prior-based Segmentation workflow that has shown optimal results for non-human primate data.

      # The atlas image assumed to be used in ANTs Prior-based Segmentation.
      template_brain_list :
        - /cpac_templates/MacaqueYerkes19_T1w_0.5mm_desc-JLC_T1w_brain.nii.gz
        - /cpac_templates/J_Macaque_11mo_atlas_nACQ_194x252x160space_0.5mm_desc-JLC_T1w_brain.nii.gz

      # The atlas segmentation images.
      # For performing ANTs Prior-based segmentation method
      # the number of specified segmentations should be identical to the number of atlas brain image sets.
      # eg.
      # ANTs_prior_seg_template_brain_list :
      #   - atlas1.nii.gz
      #   - atlas2.nii.gz
      # ANTs_prior_seg_template_segmentation_list:
      #   - segmentation1.nii.gz
      #   - segmentation1.nii.gz
      template_segmentation_list:
        - /cpac_templates/MacaqueYerkes19_T1w_0.5mm_desc-JLC_Segmentation.nii.gz
        - /cpac_templates/J_Macaque_11mo_atlas_nACQ_194x252x160space_0.5mm_desc-JLC_Segmentation.nii.gz

      # Label values corresponding to CSF/GM/WM in atlas file
      # It is not necessary to change this values unless your CSF/GM/WM label values are different from Freesurfer Color Lookup Table.
      # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

      # Label values corresponding to CSF in multiatlas file
      CSF_label : [24]

      # Label values corresponding to Gray Matter in multiatlas file
      GM_label : [3, 42]

      # Label values corresponding to White Matter in multiatlas file
      WM_label : [2, 41]

    FreeSurfer:

      # Use mri_binarize --erode option to erode segmentation masks
      erode: 0

      # Label values corresponding to CSF in FreeSurfer aseg segmentation file
      CSF_label : [24]

      # Label values corresponding to Gray Matter in FreeSurfer aseg segmentation file
      GM_label : [3, 42]

      # Label values corresponding to White Matter in FreeSurfer aseg segmentation file
      WM_label : [2, 41]


registration_workflows:

  anatomical_registration:

    run: On

    # The resolution to which anatomical images should be transformed during registration.
    # This is the resolution at which processed anatomical files will be output.
    resolution_for_anat: 2mm

    # Template to be used during registration.
    # It is not necessary to change this path unless you intend to use a non-standard template.
    T1w_brain_template: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain.nii.gz

    # Template to be used during registration.
    # It is not necessary to change this path unless you intend to use a non-standard template.
    T1w_template: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}.nii.gz

    # Template to be used during registration.
    # It is not necessary to change this path unless you intend to use a non-standard template.
    T1w_brain_template_mask: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask.nii.gz

    # Register skull-on anatomical image to a template.
    reg_with_skull: True

    registration:

      # using: ['ANTS', 'FSL', 'FSL-linear']
      # this is a fork point
      #   selecting both ['ANTS', 'FSL'] will run both and fork the pipeline
      using: ['ANTS']

      # option parameters
      ANTs:

        # If a lesion mask is available for a T1w image, use it to improve the ANTs' registration
        # ANTS registration only.
        use_lesion_mask: False

        # ANTs parameters for T1-template-based registration
        T1_registration:

          - collapse-output-transforms: 0
          - dimensionality: 3
          - initial-moving-transform :
             initializationFeature: 0

          - transforms:
             - Rigid:
                 gradientStep : 0.1
                 metric :
                   type : MI
                   metricWeight: 1
                   numberOfBins : 32
                   samplingStrategy : Regular
                   samplingPercentage : 0.25
                 convergence:
                   iteration : 1000x500x250x100
                   convergenceThreshold : 1e-08
                   convergenceWindowSize : 10
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 8x4x2x1
                 use-histogram-matching : True

             - Affine:
                 gradientStep : 0.1
                 metric :
                   type : MI
                   metricWeight: 1
                   numberOfBins : 32
                   samplingStrategy : Regular
                   samplingPercentage : 0.25
                 convergence:
                   iteration : 1000x500x250x100
                   convergenceThreshold : 1e-08
                   convergenceWindowSize : 10
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 8x4x2x1
                 use-histogram-matching : True

             - SyN:
                 gradientStep : 0.1
                 updateFieldVarianceInVoxelSpace : 3.0
                 totalFieldVarianceInVoxelSpace : 0.0
                 metric:
                   type : CC
                   metricWeight: 1
                   radius : 4
                 convergence:
                   iteration : 100x100x70x20
                   convergenceThreshold : 1e-09
                   convergenceWindowSize : 15
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 6x4x2x1
                 use-histogram-matching : True
                 winsorize-image-intensities :
                   lowerQuantile : 0.01
                   upperQuantile : 0.99

        # Interpolation method for writing out transformed anatomical images.
        # Possible values: Linear, BSpline, LanczosWindowedSinc
        interpolation: LanczosWindowedSinc

      FSL-FNIRT:

        # Configuration file to be used by FSL to set FNIRT parameters.
        # It is not necessary to change this path unless you intend to use custom FNIRT parameters or a non-standard template.
        fnirt_config: T1_2_MNI152_2mm

        # The resolution to which anatomical images should be transformed during registration.
        # This is the resolution at which processed anatomical files will be output. 
        # specifically for monkey pipeline
        ref_resolution: 2mm

        # Reference mask for FSL registration.
        ref_mask: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask_dil.nii.gz
        
        # Template to be used during registration.
        # It is for monkey pipeline specifically. 
        FNIRT_T1w_brain_template: None

        # Template to be used during registration.
        # It is for monkey pipeline specifically. 
        FNIRT_T1w_template: None
        
        # Interpolation method for writing out transformed anatomical images.
        # Possible values: trilinear, sinc, spline
        interpolation: sinc

        # Identity matrix used during FSL-based resampling of anatomical-space data throughout the pipeline.
        # It is not necessary to change this path unless you intend to use a different template.
        identity_matrix: /usr/share/fsl/5.0/etc/flirtsch/ident.mat

        # Reference mask for FSL registration.
        ref_mask: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask_dil.nii.gz

        # Reference mask with 2mm resolution to be used during FNIRT-based brain extraction in ABCD-options pipeline.
        ref_mask_res-2: /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz

        # Template with 2mm resolution to be used during FNIRT-based brain extraction in ABCD-options pipeline.
        T1w_template_res-2: /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz

    overwrite_transform:

      run: Off

      # Choose the tool to overwrite transform, currently only support 'FSL' to overwrite 'ANTs' transforms in ABCD-options pipeline.
      # using: 'FSL'
      using: FSL

  functional_registration:

    coregistration:
      # functional (BOLD/EPI) registration to anatomical (structural/T1)

      run: On

      # reference: 'brain' or 'restore-brain'
      # In ABCD-options pipeline, 'restore-brain' is used as coregistration reference
      reference: brain

      # Choose FSL or ABCD as coregistration method
      using: FSL

      # Choose brain or whole-head as coregistration input
      input: brain

      # Choose coregistration interpolation
      interpolation: trilinear

      # Choose coregistration cost function
      cost: corratio

      # Choose coregistration degree of freedom
      dof: 6

      # Extra arguments for FSL flirt
      arguments: None

      func_input_prep:

        # Choose whether to use functional brain or skull as the input to functional-to-anatomical registration
        reg_with_skull: Off

        # Choose whether to use the mean of the functional/EPI as the input to functional-to-anatomical registration or one of the volumes from the functional 4D timeseries that you choose.
        # input: ['Mean_Functional', 'Selected_Functional_Volume', 'fmriprep_reference']
        input: ['Mean_Functional']

        Mean Functional:

          # Run ANTs’ N4 Bias Field Correction on the input BOLD (EPI)
          # this can increase tissue contrast which may improve registration quality in some data
          n4_correct_func: False

        Selected Functional Volume:

          # Only for when 'Use as Functional-to-Anatomical Registration Input' is set to 'Selected Functional Volume'.
          #Input the index of which volume from the functional 4D timeseries input file you wish to use as the input for functional-to-anatomical registration.
          func_reg_input_volume: 0

      boundary_based_registration:
        # this is a fork point
        #   run: [On, Off] - this will run both and fork the pipeline
        run: [On]

        # Standard FSL 5.0 Scheduler used for Boundary Based Registration.
        # It is not necessary to change this path unless you intend to use non-standard MNI registration.
        bbr_schedule: /usr/share/fsl/5.0/etc/flirtsch/bbr.sch

        # reference for boundary based registration
        # options: 'whole-head' or 'brain'
        reference: whole-head

        # choose which FAST map to generate BBR WM mask
        # options: 'probability_map', 'partial_volume_map'
        bbr_wm_map: 'probability_map'

        # optional FAST arguments to generate BBR WM mask
        bbr_wm_mask_args: '-thr 0.5 -bin'

    EPI_registration:

      # directly register the mean functional to an EPI template
      #   instead of applying the anatomical T1-to-template transform to the functional data that has been
      #   coregistered to anatomical/T1 space
      run: Off

      # using: ['ANTS', 'FSL', 'FSL-linear']
      # this is a fork point
      # ex. selecting both ['ANTS', 'FSL'] will run both and fork the pipeline
      using: ['ANTS']

      # EPI template for direct functional-to-template registration
      # (bypassing coregistration and the anatomical-to-template transforms)
      EPI_template: s3://fcp-indi/resources/cpac/resources/epi_hbn.nii.gz

      # EPI template mask.
      EPI_template_mask: None

      ANTs:

        # EPI registration configuration - synonymous with T1_registration
        # parameters under anatomical registration above
        parameters:

          - collapse-output-transforms: 0
          - dimensionality: 3
          - initial-moving-transform :
             initializationFeature: 0

          - transforms:
             - Rigid:
                 gradientStep : 0.1
                 metric :
                   type : MI
                   metricWeight: 1
                   numberOfBins : 32
                   samplingStrategy : Regular
                   samplingPercentage : 0.25
                 convergence:
                   iteration : 1000x500x250x100
                   convergenceThreshold : 1e-08
                   convergenceWindowSize : 10
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 8x4x2x1
                 use-histogram-matching : True

             - Affine:
                 gradientStep : 0.1
                 metric :
                   type : MI
                   metricWeight: 1
                   numberOfBins : 32
                   samplingStrategy : Regular
                   samplingPercentage : 0.25
                 convergence:
                   iteration : 1000x500x250x100
                   convergenceThreshold : 1e-08
                   convergenceWindowSize : 10
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 8x4x2x1
                 use-histogram-matching : True

             - SyN:
                 gradientStep : 0.1
                 updateFieldVarianceInVoxelSpace : 3.0
                 totalFieldVarianceInVoxelSpace : 0.0
                 metric:
                   type : CC
                   metricWeight: 1
                   radius : 4
                 convergence:
                   iteration : 100x100x70x20
                   convergenceThreshold : 1e-09
                   convergenceWindowSize : 15
                 smoothing-sigmas : 3.0x2.0x1.0x0.0
                 shrink-factors : 6x4x2x1
                 use-histogram-matching : True
                 winsorize-image-intensities :
                   lowerQuantile : 0.01
                   upperQuantile : 0.99

        # Interpolation method for writing out transformed EPI images.
        # Possible values: Linear, BSpline, LanczosWindowedSinc
        interpolation: LanczosWindowedSinc

      FSL-FNIRT:

        # Configuration file to be used by FSL to set FNIRT parameters.
        # It is not necessary to change this path unless you intend to use custom FNIRT parameters or a non-standard template.
        fnirt_config: T1_2_MNI152_2mm

        # Interpolation method for writing out transformed EPI images.
        # Possible values: trilinear, sinc, spline
        interpolation: sinc

        # Identity matrix used during FSL-based resampling of BOLD-space data throughout the pipeline.
        # It is not necessary to change this path unless you intend to use a different template.
        identity_matrix: /usr/share/fsl/5.0/etc/flirtsch/ident.mat

    func_registration_to_template:

      # these options modify the application (to the functional data), not the calculation, of the
      # T1-to-template and EPI-to-template transforms calculated earlier during registration
      
      # apply the functional-to-template (T1 template) registration transform to the functional data
      run: On
      
      # apply the functional-to-template (EPI template) registration transform to the functional data
      run_EPI: Off

      output_resolution:

        # The resolution (in mm) to which the preprocessed, registered functional timeseries outputs are written into.
        # NOTE:
        #   selecting a 1 mm or 2 mm resolution might substantially increase your RAM needs- these resolutions should be selected with caution.
        #   for most cases, 3 mm or 4 mm resolutions are suggested.
        # NOTE:
        #   this also includes the single-volume 3D preprocessed functional data,
        #   such as the mean functional (mean EPI) in template space
        func_preproc_outputs: 3mm

        # The resolution (in mm) to which the registered derivative outputs are written into.
        # NOTE:
        #   this is for the single-volume functional-space outputs (i.e. derivatives)
        #   thus, a higher resolution may not result in a large increase in RAM needs as above
        func_derivative_outputs: 3mm

      target_template:      
        # choose which template space to transform derivatives towards
        # using: ['T1_template', 'EPI_template']
        # this is a fork point
        # NOTE:
        #   this will determine which registration transform to use to warp the functional
        #   outputs and derivatives to template space
        using: ['T1_template']

        T1_template:

          # Standard Skull Stripped Template. Used as a reference image for functional registration.
          # This can be different than the template used as the reference/fixed for T1-to-template registration.
          T1w_brain_template_funcreg: /usr/share/fsl/5.0/data/standard/MNI152_T1_${func_resolution}_brain.nii.gz

          # Standard Anatomical Brain Image with Skull.
          # This can be different than the template used as the reference/fixed for T1-to-template registration.
          T1w_template_funcreg: /usr/share/fsl/5.0/data/standard/MNI152_T1_${func_resolution}.nii.gz

          # Template to be used during registration.
          # It is not necessary to change this path unless you intend to use a non-standard template.
          T1w_brain_template_mask_funcreg: /usr/share/fsl/5.0/data/standard/MNI152_T1_${func_resolution}_brain_mask.nii.gz

          # a standard template for resampling if using float resolution
          T1w_template_for_resample:  $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz

        EPI_template:

          # EPI template for direct functional-to-template registration
          # (bypassing coregistration and the anatomical-to-template transforms)
          EPI_template_funcreg: s3://fcp-indi/resources/cpac/resources/epi_hbn.nii.gz

          # EPI template mask.
          EPI_template_mask_funcreg: None

          # a standard template for resampling if using float resolution
          EPI_template_for_resample:  $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz

      ANTs_pipelines:

        # Interpolation method for writing out transformed functional images.
        # Possible values: Linear, BSpline, LanczosWindowedSinc
        interpolation: LanczosWindowedSinc

      FNIRT_pipelines:

        # Interpolation method for writing out transformed functional images.
        # Possible values: trilinear, sinc, spline
        interpolation: sinc

        # Identity matrix used during FSL-based resampling of functional-space data throughout the pipeline.
        # It is not necessary to change this path unless you intend to use a different template.
        identity_matrix: /usr/share/fsl/5.0/etc/flirtsch/ident.mat

      apply_transform:

        # options: 'default', 'abcd', 'single_step_resampling', 'dcan_nhp'
        # 'default': apply func-to-anat and anat-to-template transforms on motion corrected functional image.
        # 'single_step_resampling': apply motion correction, func-to-anat and anat-to-template transforms on each of PREPROCESSED functional volume using ANTs antsApplyTransform based on fMRIPrep pipeline. 
        # 'abcd': apply motion correction, func-to-anat and anat-to-template transforms on each of raw functional volume using FSL applywarp based on ABCD-HCP pipeline.
        # 'single_step_resampling_from_stc': apply motion correction, func-to-anat and anat-to-template transforms on each of slice-time-corrected functional volume using ANTs antsApplyTransform based on fMRIPrep pipeline.
        using: 'default'


functional_preproc:

  run: On

  truncation:

    # First timepoint to include in analysis.
    # Default is 0 (beginning of timeseries).
    # First timepoint selection in the scan parameters in the data configuration file, if present, will over-ride this selection.
    # Note: the selection here applies to all scans of all participants.
    start_tr: 0

    # Last timepoint to include in analysis.
    # Default is None or End (end of timeseries).
    # Last timepoint selection in the scan parameters in the data configuration file, if present, will over-ride this selection.
    # Note: the selection here applies to all scans of all participants.
    stop_tr: None

  scaling:

    # Scale functional raw data, usually used in rodent pipeline
    run: Off

    # Scale the size of the dataset voxels by the factor.
    scaling_factor: 10

  despiking:

    # Run AFNI 3dDespike
    # this is a fork point
    #   run: [On, Off] - this will run both and fork the pipeline
    run: [Off]

  slice_timing_correction:

    # Interpolate voxel time courses so they are sampled at the same time points.
    # this is a fork point
    #   run: [On, Off] - this will run both and fork the pipeline
    run: [On]

    # use specified slice time pattern rather than one in header
    tpattern: None

    # align each slice to given time offset
    # The default alignment time is the average of the 'tpattern' values (either from the dataset header or from the tpattern option).
    tzero: None

  motion_estimates_and_correction:
  
    run: On

    motion_estimates: 

      # calculate motion statistics BEFORE slice-timing correction
      calculate_motion_first: Off

      # calculate motion statistics AFTER motion correction
      calculate_motion_after: On

    motion_correction:

      # using: ['3dvolreg', 'mcflirt']
      # this is a fork point
      using: ['3dvolreg']

      # option parameters
      AFNI-3dvolreg:

        # This option is useful when aligning high-resolution datasets that may need more alignment than a few voxels.
        functional_volreg_twopass: On

      # Choose motion correction reference. Options: mean, median, selected_volume, fmriprep_reference
      motion_correction_reference: ['mean']

      # Choose motion correction reference volume
      motion_correction_reference_volume: 0

    motion_estimate_filter:

      # Filter physiological (respiration) artifacts from the head motion estimates.
      # Adapted from DCAN Labs filter.
      #     https://www.ohsu.edu/school-of-medicine/developmental-cognition-and-neuroimaging-lab
      #     https://www.biorxiv.org/content/10.1101/337360v1.full.pdf
      # this is a fork point
      #   run: [On, Off] - this will run both and fork the pipeline
      run: [Off]

      # options: "notch", "lowpass"
      filter_type: "notch"

      # Number of filter coefficients.
      filter_order: 4

      # Dataset-wide respiratory rate data from breathing belt.
      # Notch filter requires either:
      #     "breathing_rate_min" and "breathing_rate_max"
      # or
      #     "center_frequency" and "filter_bandwitdh".
      # Lowpass filter requires either:
      #     "breathing_rate_min"
      # or
      #     "lowpass_cutoff".
      # If "breathing_rate_min" (for lowpass and notch filter)
      # and "breathing_rate_max" (for notch filter) are set,
      # the values set in "lowpass_cutoff" (for lowpass filter),
      # "center_frequency" and "filter_bandwidth" (for notch filter)
      # options are ignored.

      # Lowest Breaths-Per-Minute in dataset.
      # For both notch and lowpass filters.
      breathing_rate_min:

      # Highest Breaths-Per-Minute in dataset.
      # For notch filter.
      breathing_rate_max:

      # notch filter direct customization parameters

      # mutually exclusive with breathing_rate options above.
      # If breathing_rate_min and breathing_rate_max are provided,
      # the following parameters will be ignored.

      # the center frequency of the notch filter
      center_frequency:

      # the width of the notch filter
      filter_bandwidth:

      # lowpass filter direct customization parameter

      # mutually exclusive with breathing_rate options above.
      # If breathing_rate_min is provided, the following
      # parameter will be ignored.

      # the frequency cutoff of the filter
      lowpass_cutoff:

  distortion_correction:

    # this is a fork point
    #   run: [On, Off] - this will run both and fork the pipeline
    run: [On]

    # using: ['PhaseDiff', 'Blip', 'Blip-FSL-TOPUP']
    #   PhaseDiff - Perform field map correction using a single phase difference image, a subtraction of the two phase images from each echo. Default scanner for this method is SIEMENS.
    #   Blip - Uses AFNI 3dQWarp to calculate the distortion unwarp for EPI field maps of opposite/same phase encoding direction.
    #   Blip-FSL-TOPUP - Uses FSL TOPUP to calculate the distortion unwarp for EPI field maps of opposite/same phase encoding direction.
    using: ['PhaseDiff', 'Blip']

    # option parameters
    PhaseDiff:

      # Since the quality of the distortion heavily relies on the skull-stripping step, we provide a choice of method ('AFNI' for AFNI 3dSkullStrip or 'BET' for FSL BET).
      # Options: 'BET' or 'AFNI'
      fmap_skullstrip_option: 'BET'

      # Set the fraction value for the skull-stripping of the magnitude file. Depending on the data, a tighter extraction may be necessary in order to prevent noisy voxels from interfering with preparing the field map.
      # The default value is 0.5.
      fmap_skullstrip_BET_frac: 0.5

      # Set the threshold value for the skull-stripping of the magnitude file. Depending on the data, a tighter extraction may be necessary in order to prevent noisy voxels from interfering with preparing the field map.
      # The default value is 0.6.
      fmap_skullstrip_AFNI_threshold:  0.6
      
    Blip-FSL-TOPUP:
      
      # (approximate) resolution (in mm) of warp basis for the different sub-sampling levels, default 10
      warpres: 10
      
      # sub-sampling scheme, default 1
      subsamp: 1
      
      # FWHM (in mm) of gaussian smoothing kernel, default 8
      fwhm: 8
      
      # Max # of non-linear iterations, default 5
      miter: 5
      
      # Weight of regularisation, default depending on --ssqlambda and --regmod switches. See user documentation.
      lambda: 1
      
      # If set (=1), lambda is weighted by current ssq, default 1
      ssqlambda: 1
      
      # Model for regularisation of warp-field [membrane_energy bending_energy], default bending_energy
      regmod: bending_energy
      
      # Estimate movements if set, default 1 (true)
      estmov: 1
      
      # Minimisation method 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient, default 0 (LM)
      minmet: 0
      
      # Order of spline, 2->Qadratic spline, 3->Cubic spline. Default=3
      splineorder: 3
      
      # Precision for representing Hessian, double or float. Default double
      numprec: double
      
      # Image interpolation model, linear or spline. Default spline
      interp: spline
      
      # If set (=1), the images are individually scaled to a common mean, default 0 (false)
      scale: 0
      
      # If set (=1), the calculations are done in a different grid, default 1 (true)
      regrid: 1

  func_masking:

    # using: ['AFNI', 'FSL', 'FSL_AFNI', 'Anatomical_Refined', 'Anatomical_Based', 'Anatomical_Resampled', 'CCS_Anatomical_Refined']

    # FSL_AFNI: fMRIPrep-style BOLD mask. Ref: https://github.com/nipreps/niworkflows/blob/a221f612/niworkflows/func/util.py#L246-L514
    # Anatomical_Refined: 1. binarize anat mask, in case it is not a binary mask. 2. fill holes of anat mask 3. init_bold_mask : input raw func → dilate init func brain mask 4. refined_bold_mask : input motion corrected func → dilate anatomical mask 5. get final func mask
    # Anatomical_Based: Generate the BOLD mask by basing it off of the anatomical brain mask. Adapted from DCAN Lab's BOLD mask method from the ABCD pipeline.
    # Anatomical_Resampled: Resample anatomical brain mask in standard space to get BOLD brain mask in standard space. Adapted from DCAN Lab's BOLD mask method from the ABCD pipeline. ("Create fMRI resolution standard space files for T1w image, wmparc, and brain mask […] don't use FLIRT to do spline interpolation with -applyisoxfm for the 2mm and 1mm cases because it doesn't know the peculiarities of the MNI template FOVs")
    # CCS_Anatomical_Refined: Generate the BOLD mask by basing it off of the anatomical brain. Adapted from the BOLD mask method from the CCS pipeline.

    # this is a fork point
    using: ['AFNI']

    FSL-BET:

      # Apply to 4D FMRI data, if bold_bet_functional_mean_boolean : Off.
      # Mutually exclusive with functional, reduce_bias, robust, padding, remove_eyes, surfaces
      # It must be 'on' if select 'reduce_bias', 'robust', 'padding', 'remove_eyes', or 'bet_surfaces' on
      functional_mean_boolean: Off

      # Set an intensity threshold to improve skull stripping performances of FSL BET on rodent scans.
      functional_mean_thr: 
        run: Off
        threshold_value: 98

      # Bias correct the functional mean image to improve skull stripping performances of FSL BET on rodent scans
      functional_mean_bias_correction: Off

      # Set the threshold value controling the brain vs non-brain voxels.
      frac: 0.3

      # Mesh created along with skull stripping
      mesh_boolean: Off

      # Create a surface outline image
      outline: Off

      # Add padding to the end of the image, improving BET.Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      padding: Off

      # Integer value of head radius
      radius: 0

      # Reduce bias and cleanup neck. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      reduce_bias: Off

      # Eyes and optic nerve cleanup. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      remove_eyes: Off

      # Robust brain center estimation. Mutually exclusive with functional,reduce_bias,robust,padding,remove_eyes,surfaces
      robust: Off

      # Create a skull image
      skull: Off

      # Gets additional skull and scalp surfaces by running bet2 and betsurf. This is mutually exclusive with reduce_bias, robust, padding, remove_eyes
      surfaces: Off

      # Apply thresholding to segmented brain image and mask
      threshold: Off

      # Vertical gradient in fractional intensity threshold (-1,1)
      vertical_gradient: 0.0

    FSL_AFNI:

      bold_ref:

      brain_mask: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask.nii.gz

      brain_probseg: /usr/share/fsl/5.0/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask.nii.gz

    Anatomical_Refined:

      # Choose whether or not to dilate the anatomical mask if you choose 'Anatomical_Refined' as the functional masking option. It will dilate one voxel if enabled.
      anatomical_mask_dilation: False

    # Apply functional mask in native space
    apply_func_mask_in_native_space: On

  generate_func_mean:

    # Generate mean functional image
    run: On

  normalize_func:

    # Normalize functional image
    run: On


nuisance_corrections:

  1-ICA-AROMA:

    # this is a fork point
    #   run: [On, Off] - this will run both and fork the pipeline
    run: [Off]

    # Types of denoising strategy:
    #   nonaggr: nonaggressive-partial component regression
    #   aggr:    aggressive denoising
    denoising_type: nonaggr

  2-nuisance_regression:

    # this is a fork point
    #   run: [On, Off] - this will run both and fork the pipeline
    run: [On]

    # switch to Off if nuisance regression is off and you don't want to write out the regressors
    create_regressors: On

    # Select which nuisance signal corrections to apply
    Regressors:

      -  Name: 'default'

         Motion:
           include_delayed: true
           include_squared: true
           include_delayed_squared: true

         CerebrospinalFluid:
           summary: Mean
           extraction_resolution: 2
           erode_mask: true

         GlobalSignal:
           summary: Mean

         PolyOrt:
          degree: 2

         Bandpass:
           bottom_frequency: 0.01
           top_frequency: 0.1
           method: default

      -  Name: 'defaultNoGSR'

         Motion:
           include_delayed: true
           include_squared: true
           include_delayed_squared: true

         aCompCor:
           summary:
             method: DetrendPC
             components: 5
           tissues:
             - WhiteMatter
             - CerebrospinalFluid
           extraction_resolution: 2

         CerebrospinalFluid:
           summary: Mean
           extraction_resolution: 2
           erode_mask: true

         PolyOrt:
          degree: 2

         Bandpass:
           bottom_frequency: 0.01
           top_frequency: 0.1
           method: default

    # Standard Lateral Ventricles Binary Mask
    # used in CSF mask refinement for CSF signal-related regressions
    lateral_ventricles_mask: $FSLDIR/data/atlases/HarvardOxford/HarvardOxford-lateral-ventricles-thr25-2mm.nii.gz

    # Whether to run frequency filtering before or after nuisance regression.
    # Options: 'After' or 'Before'
    bandpass_filtering_order: 'After'

    # Process and refine masks used to produce regressors and time series for
    # regression.
    regressor_masks:

      erode_anatomical_brain_mask:

        # Erode binarized anatomical brain mask. If choosing True, please also set regressor_masks['erode_csf']['run']: True; anatomical_preproc['brain_extraction']['using']: niworkflows-ants.
        run: Off

        # Target volume ratio, if using erosion.
        # Default proportion is None for anatomical brain mask.
        # If using erosion, using both proportion and millimeters is not recommended.
        brain_mask_erosion_prop:

        # Erode brain mask in millimeters, default for brain mask is 30 mm
        # Brain erosion default is using millimeters.
        brain_mask_erosion_mm: 30

        # Erode binarized brain mask in millimeter
        brain_erosion_mm:

      erode_csf:

        # Erode binarized csf tissue mask.
        run: Off

        # Target volume ratio, if using erosion.
        # Default proportion is None for cerebrospinal fluid mask.
        # If using erosion, using both proportion and millimeters is not recommended.
        csf_erosion_prop:

        # Erode cerebrospinal fluid mask in millimeters, default for cerebrospinal fluid is 30mm
        # Cerebrospinal fluid erosion default is using millimeters.
        csf_mask_erosion_mm: 30

        # Erode binarized cerebrospinal fluid mask in millimeter
        csf_erosion_mm:

      erode_wm:

        # Erode WM binarized tissue mask.
        run: Off

        # Target volume ratio, if using erosion.
        # Default proportion is 0.6 for white matter mask.
        # If using erosion, using both proportion and millimeters is not recommended.
        # White matter erosion default is using proportion erosion method when use erosion for white matter.
        wm_erosion_prop: 0.6

        # Erode white matter mask in millimeters, default for white matter is None
        wm_mask_erosion_mm:

        # Erode binarized white matter mask in millimeters
        wm_erosion_mm:

      erode_gm:

        # Erode gray matter binarized tissue mask.
        run: Off

        # Target volume ratio, if using erosion.
        # If using erosion, using both proportion and millimeters is not recommended.
        gm_erosion_prop: 0.6

        # Erode gray matter mask in millimeters
        gm_mask_erosion_mm:

        # Erode binarized gray matter mask in millimeters
        gm_erosion_mm:


# OUTPUTS AND DERIVATIVES
# -----------------------
post_processing:

  spatial_smoothing:

    # Smooth the derivative outputs.
    # Set as ['nonsmoothed'] to disable smoothing. Set as ['smoothed', 'nonsmoothed'] to get both.
    #
    # Options:
    #     ['smoothed', 'nonsmoothed']
    output: ['smoothed']

    # Tool to use for smoothing.
    # 'FSL' for FSL MultiImageMaths for FWHM provided
    # 'AFNI' for AFNI 3dBlurToFWHM for FWHM provided
    smoothing_method: ['FSL']

    # Full Width at Half Maximum of the Gaussian kernel used during spatial smoothing.
    # this is a fork point
    #   i.e. multiple kernels - fwhm: [4,6,8]
    fwhm: [4]

  z-scoring:

    # z-score standardize the derivatives. This may be needed for group-level analysis.
    # Set as ['raw'] to disable z-scoring. Set as ['z-scored', 'raw'] to get both.
    #
    # Options:
    #     ['z-scored', 'raw']
    output: ['z-scored']


timeseries_extraction:

  run: On

  # Enter paths to region-of-interest (ROI) NIFTI files (.nii or .nii.gz) to be used for time-series extraction, and then select which types of analyses to run.
  # Denote which analyses to run for each ROI path by listing the names below. For example, if you wish to run Avg and SpatialReg, you would enter: '/path/to/ROI.nii.gz': Avg, SpatialReg
  # available analyses:
  #   /path/to/atlas.nii.gz: Avg, Voxel, SpatialReg
  tse_roi_paths:
    /cpac_templates/CC400.nii.gz: Avg
    /cpac_templates/aal_mask_pad.nii.gz: Avg
    /cpac_templates/CC200.nii.gz: Avg
    /cpac_templates/tt_mask_pad.nii.gz: Avg
    /cpac_templates/PNAS_Smith09_rsn10.nii.gz: SpatialReg
    /cpac_templates/ho_mask_pad.nii.gz: Avg
    /cpac_templates/rois_3mm.nii.gz: Avg
    /ndmg_atlases/label/Human/AAL_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/CAPRSC_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/DKT_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/DesikanKlein_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/HarvardOxfordcort-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/HarvardOxfordsub-maxprob-thr25_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Juelich_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/MICCAI_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Schaefer1000_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Schaefer200_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Schaefer300_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Schaefer400_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Talairach_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Brodmann_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Desikan_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Glasser_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Slab907_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Yeo-17-liberal_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Yeo-17_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Yeo-7-liberal_space-MNI152NLin6_res-1x1x1.nii.gz: Avg
    /ndmg_atlases/label/Human/Yeo-7_space-MNI152NLin6_res-1x1x1.nii.gz: Avg

  # Functional time-series and ROI realignment method: ['ROI_to_func'] or ['func_to_ROI']
  # 'ROI_to_func' will realign the atlas/ROI to functional space (fast)
  # 'func_to_ROI' will realign the functional time series to the atlas/ROI space
  #
  #     NOTE: in rare cases, realigning the ROI to the functional space may
  #           result in small misalignments for very small ROIs - please double
  #           check your data if you see issues
  realignment: 'ROI_to_func'

  connectivity_matrix:
    # Create a connectivity matrix from timeseries data

    # Options:
    #  ['AFNI', 'Nilearn', 'ndmg']
    using:
      - Nilearn
      - ndmg
    # Options:
    #  ['Pearson', 'Partial']
    # Note: These options are not configurable for ndmg, which will ignore these options
    measure:
      - Pearson
      - Partial


seed_based_correlation_analysis:

  # SCA - Seed-Based Correlation Analysis
  # For each extracted ROI Average time series, CPAC will generate a whole-brain correlation map.
  # It should be noted that for a given seed/ROI, SCA maps for ROI Average time series will be the same.
  run: Off

  # Enter paths to region-of-interest (ROI) NIFTI files (.nii or .nii.gz) to be used for seed-based correlation analysis, and then select which types of analyses to run.
  # Denote which analyses to run for each ROI path by listing the names below. For example, if you wish to run Avg and MultReg, you would enter: '/path/to/ROI.nii.gz': Avg, MultReg
  # available analyses:
  #   /path/to/atlas.nii.gz: Avg, DualReg, MultReg
  sca_roi_paths:
    /cpac_templates/PNAS_Smith09_rsn10.nii.gz: DualReg
    /cpac_templates/CC400.nii.gz: Avg, MultReg
    /cpac_templates/ez_mask_pad.nii.gz: Avg, MultReg
    /cpac_templates/aal_mask_pad.nii.gz: Avg, MultReg
    /cpac_templates/CC200.nii.gz: Avg, MultReg
    /cpac_templates/tt_mask_pad.nii.gz: Avg, MultReg
    /cpac_templates/ho_mask_pad.nii.gz: Avg, MultReg
    /cpac_templates/rois_3mm.nii.gz: Avg, MultReg

  # Normalize each time series before running Dual Regression SCA.
  norm_timeseries_for_DR: True


amplitude_low_frequency_fluctuation:

  # ALFF & f/ALFF
  # Calculate Amplitude of Low Frequency Fluctuations (ALFF) and fractional ALFF (f/ALFF) for all voxels.
  run: On

  # Frequency cutoff (in Hz) for the high-pass filter used when calculating f/ALFF.
  highpass_cutoff: [0.01]

  # Frequency cutoff (in Hz) for the low-pass filter used when calculating f/ALFF
  lowpass_cutoff: [0.1]


regional_homogeneity:

  # ReHo
  # Calculate Regional Homogeneity (ReHo) for all voxels.
  run: On

  # Number of neighboring voxels used when calculating ReHo
  # 7 (Faces)
  # 19 (Faces + Edges)
  # 27 (Faces + Edges + Corners)
  cluster_size: 27


voxel_mirrored_homotopic_connectivity:

  # VMHC
  # Calculate Voxel-mirrored Homotopic Connectivity (VMHC) for all voxels.
  run: On

  symmetric_registration:

    # Included as part of the 'Image Resource Files' package available on the Install page of the User Guide.
    # It is not necessary to change this path unless you intend to use a non-standard symmetric template.
    T1w_brain_template_symmetric: $FSLDIR/data/standard/MNI152_T1_${resolution_for_anat}_brain_symmetric.nii.gz

    # A reference symmetric brain template for resampling
    T1w_brain_template_symmetric_for_resample: $FSLDIR/data/standard/MNI152_T1_1mm_brain_symmetric.nii.gz

    # Included as part of the 'Image Resource Files' package available on the Install page of the User Guide.
    # It is not necessary to change this path unless you intend to use a non-standard symmetric template.
    T1w_template_symmetric: $FSLDIR/data/standard/MNI152_T1_${resolution_for_anat}_symmetric.nii.gz

    # A reference symmetric skull template for resampling
    T1w_template_symmetric_for_resample: $FSLDIR/data/standard/MNI152_T1_1mm_symmetric.nii.gz

    # Included as part of the 'Image Resource Files' package available on the Install page of the User Guide.
    # It is not necessary to change this path unless you intend to use a non-standard symmetric template.
    dilated_symmetric_brain_mask: $FSLDIR/data/standard/MNI152_T1_${resolution_for_anat}_brain_mask_symmetric_dil.nii.gz

    # A reference symmetric brain mask template for resampling
    dilated_symmetric_brain_mask_for_resample: $FSLDIR/data/standard/MNI152_T1_1mm_brain_mask_symmetric_dil.nii.gz


network_centrality:

  # Calculate Degree, Eigenvector Centrality, or Functional Connectivity Density.
  run: On

  # Maximum amount of RAM (in GB) to be used when calculating Degree Centrality.
  # Calculating Eigenvector Centrality will require additional memory based on the size of the mask or number of ROI nodes.
  memory_allocation:  1.0

  # Full path to a NIFTI file describing the mask. Centrality will be calculated for all voxels within the mask.
  template_specification_file:  /cpac_templates/Mask_ABIDE_85Percent_GM.nii.gz

  degree_centrality:

    # Enable/Disable degree centrality by selecting the connectivity weights
    #   weight_options: ['Binarized', 'Weighted']
    # disable this type of centrality with:
    #   weight_options: []
    weight_options:  ['Binarized', 'Weighted']

    # Select the type of threshold used when creating the degree centrality adjacency matrix.
    # options:
    #   'Significance threshold', 'Sparsity threshold', 'Correlation threshold'
    correlation_threshold_option: 'Sparsity threshold'

    # Based on the Threshold Type selected above, enter a Threshold Value.
    # P-value for Significance Threshold
    # Sparsity value for Sparsity Threshold
    # Pearson's r value for Correlation Threshold
    correlation_threshold: 0.001

  eigenvector_centrality:

    # Enable/Disable eigenvector centrality by selecting the connectivity weights
    #   weight_options: ['Binarized', 'Weighted']
    # disable this type of centrality with:
    #   weight_options: []
    weight_options: ['Weighted']

    # Select the type of threshold used when creating the eigenvector centrality adjacency matrix.
    # options:
    #   'Significance threshold', 'Sparsity threshold', 'Correlation threshold'
    correlation_threshold_option: 'Sparsity threshold'

    # Based on the Threshold Type selected above, enter a Threshold Value.
    # P-value for Significance Threshold
    # Sparsity value for Sparsity Threshold
    # Pearson's r value for Correlation Threshold
    correlation_threshold: 0.001

  local_functional_connectivity_density:

    # Enable/Disable lFCD by selecting the connectivity weights
    #   weight_options: ['Binarized', 'Weighted']
    # disable this type of centrality with:
    #   weight_options: []
    weight_options: ['Binarized', 'Weighted']

    # Select the type of threshold used when creating the lFCD adjacency matrix.
    # options:
    #   'Significance threshold', 'Correlation threshold'
    correlation_threshold_option: 'Correlation threshold'

    # Based on the Threshold Type selected above, enter a Threshold Value.
    # P-value for Significance Threshold
    # Sparsity value for Sparsity Threshold
    # Pearson's r value for Correlation Threshold
    correlation_threshold: 0.6


# PACKAGE INTEGRATIONS
# --------------------
PyPEER:

  # Training of eye-estimation models. Commonly used for movies data/naturalistic viewing.
  run: Off

  # PEER scan names to use for training
  # Example: ['peer_run-1', 'peer_run-2']
  eye_scan_names: []

  # Naturalistic viewing data scan names to use for eye estimation
  # Example: ['movieDM']
  data_scan_names: []

  # Template-space eye mask
  eye_mask_path: $FSLDIR/data/standard/MNI152_T1_${func_resolution}_eye_mask.nii.gz

  # PyPEER Stimulus File Path
  # This is a file describing the stimulus locations from the calibration sequence.
  stimulus_path: None

  minimal_nuisance_correction:

    # PyPEER Minimal nuisance regression
    # Note: PyPEER employs minimal preprocessing - these choices do not reflect what runs in the main pipeline.
    #       PyPEER uses non-nuisance-regressed data from the main pipeline.

    # Global signal regression (PyPEER only)
    peer_gsr: True

    # Motion scrubbing (PyPEER only)
    peer_scrub: False

    # Motion scrubbing threshold (PyPEER only)
    scrub_thresh: 0.2
