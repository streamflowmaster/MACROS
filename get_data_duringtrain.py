import torch
def get_cnmr_data(peaks,test_set,
                  mask_missing_ratio=0.5, noise_range=5,
                  device = 'cpu',
                  batch_size = 16,
                  ):
    b, l = peaks.shape
    peaks = peaks.to(device).reshape(b, 2, -1)  # Assuming interleaved delta, intensity
    delta = peaks[:, 0, :]  # c_peak[:, :, 0]
    intensity = peaks[:, 1, :]  # c_peak[:, :, 1]

    # nmr_idx = [delta, intensity]
    # Create a mask to identify non-special tokens (excluding BOS=0, EOS=1, PAD=2)
    non_special_mask = torch.logical_and(
        torch.logical_and(delta != test_set.CONFIG['nmr_bos_token'],
                          delta != test_set.CONFIG['nmr_eos_token']),
        delta != test_set.CONFIG['nmr_pad_token']
    )

    # If intensity also contains special tokens, create a similar mask for it
    intensity_non_special_mask = torch.logical_and(
        torch.logical_and(intensity != test_set.CONFIG['nmr_bos_token'],
                          intensity != test_set.CONFIG['nmr_eos_token']),
        intensity != test_set.CONFIG['nmr_pad_token']
    )

    # Apply masking for missing ratio
    if mask_missing_ratio > 0:
        batch_mask = torch.rand(batch_size, device=device) < mask_missing_ratio
        cnmr_mask = batch_mask.view(-1, 1).expand_as(delta)
        # Only mask intensity where intensity_non_special_mask is True
        intensity = torch.where(cnmr_mask & intensity_non_special_mask,
                                torch.full_like(intensity, test_set.CONFIG['nmr_pad_token']),
                                intensity)

    # Apply noise to delta
    if noise_range > 0:
        # delta_noise = torch.randint(low=-noise_range, high=noise_range + 1, size=delta.shape, device=device)
        noise = torch.normal(mean=0.0, std=noise_range / 3.0, size=delta.shape, device=device)
        delta_noise = noise.round().long()
        # Only apply noise where non_special_mask is True, and clamp to avoid special tokens
        delta_pertub = torch.where(non_special_mask,
                                   torch.clamp(delta + delta_noise,
                                               min=3,  # Avoid BOS=0, EOS=1, PAD=2
                                               max=test_set.CONFIG['c_nmr_delta_disc']),
                                   # Assume NMR chemical shift upper bound
                                   delta)
        # print(delta_pertub)

        nmr_idx_pertub = [delta_pertub, intensity]
    else:
        delta_pertub = torch.where(non_special_mask,
                                   torch.clamp(delta,
                                               min=3,  # Avoid BOS=0, EOS=1, PAD=2
                                               max=test_set.CONFIG['c_nmr_delta_disc']),
                                   # Assume NMR chemical shift upper bound
                                   delta)
        nmr_idx_pertub = [delta_pertub, intensity]

    cond = None  # Replace with dataset-provided condition if available
    # print(nmr_idx_pertub)
    return nmr_idx_pertub, cond


def get_hnmr_data(peaks, test_set, mask_missing_ratio=0.5, noise_range=3, device='cpu', batch_size=16):
    b, l = peaks.shape
    peaks = peaks.to(device).reshape(b, 4, -1)  # Assuming interleaved category, centroids, jvalue, nH

    centroids = peaks[:, 0, :]  # hnmr centroids
    nH = peaks[:, 1, :]  # hnmr nH
    category = peaks[:, 2, :]  # hnmr category
    jvalue = peaks[:, 3, :]  # hnmr jvalue


    # hnmr_idx = [centroids, nH, category, jvalue]

    # Create a mask to identify non-special tokens (excluding BOS=0, EOS=1, PAD=2) for centroids
    non_special_mask = torch.logical_and(
        torch.logical_and(centroids != test_set.CONFIG['nmr_bos_token'],
                          centroids != test_set.CONFIG['nmr_eos_token']),
        centroids != test_set.CONFIG['nmr_pad_token']
    )

    # Create masks for category and jvalue, which can be replaced with PAD token
    category_non_special_mask = torch.logical_and(
        torch.logical_and(category != test_set.CONFIG['nmr_bos_token'],
                          category != test_set.CONFIG['nmr_eos_token']),
        category != test_set.CONFIG['nmr_pad_token']
    )

    jvalue_non_special_mask = torch.logical_and(
        torch.logical_and(jvalue != test_set.CONFIG['nmr_bos_token'],
                          jvalue != test_set.CONFIG['nmr_eos_token']),
        jvalue != test_set.CONFIG['nmr_pad_token']
    )

    # Apply masking for missing ratio to category and jvalue
    if mask_missing_ratio > 0:
        batch_mask = torch.rand(batch_size, device=device) < mask_missing_ratio
        hnmr_mask = batch_mask.view(-1, 1).expand_as(centroids)

        # Mask category where category_non_special_mask is True
        category = torch.where(hnmr_mask & category_non_special_mask,
                               torch.full_like(category, test_set.CONFIG['nmr_pad_token']),
                               category)

        # Mask jvalue where jvalue_non_special_mask is True
        jvalue = torch.where(hnmr_mask & jvalue_non_special_mask,
                             torch.full_like(jvalue, test_set.CONFIG['nmr_pad_token']),
                             jvalue)

    # Apply noise to centroids
    if noise_range > 0:
        # centroids_noise = torch.randint(low=-noise_range, high=noise_range + 1, size=centroids.shape, device=device)
        noise = torch.normal(0.0, noise_range / 3, size=centroids.shape, device=device)
        centroids_noise = noise.round().long()
        # Only apply noise where non_special_mask is True, and clamp to avoid special tokens
        centroids_pertub = torch.where(non_special_mask,
                                       torch.clamp(centroids + centroids_noise,
                                                   min=3,  # Avoid BOS=0, EOS=1, PAD=2
                                                   max=test_set.CONFIG['centroid_disc']),
                                       # Assume HNMR centroids upper bound
                                       centroids)

        hnmr_idx_pertub = [centroids_pertub, nH, category, jvalue]
    else:
        centroids_pertub = torch.where(non_special_mask,
                                       torch.clamp(centroids,
                                                   min=3,  # Avoid BOS=0, EOS=1, PAD=2
                                                   max=test_set.CONFIG['centroid_disc']),
                                       # Assume HNMR centroids upper bound
                                       centroids)
        hnmr_idx_pertub = [centroids_pertub, nH, category, jvalue]

    cond = None  # Replace with dataset-provided condition if available
    return hnmr_idx_pertub, cond


def get_hsqc_data(peaks, test_set, mask_missing_ratio: float = 0.0, noise_range: int = 3, device: str = 'cpu',
                  batch_size: int = 16):
    """
    Process HSQC NMR data from a loader into tensors for GPT model input.

    Args:
        loader: Data loader yielding (peaks, _) where peaks is shape (batch_size, sequence_length * 3).
        test_set: Dataset object with CONFIG containing token indices and vocab sizes.
        mask_missing_ratio (float): Fraction of non-special tokens to mask (default: 0.0, no masking).
        noise_range (int): Range for random noise added to centroids (default: 3).
        device (str): Device for output tensors ('cpu' or 'cuda').
        batch_size (int): Expected batch size (default: 16).

    Returns:
        Tuple[List[torch.Tensor], Optional[torch.Tensor]]: List of tensors [c13_centroid, h1_centroid, nh]
            each of shape (batch_size, sequence_length), and conditioning tensor (or None).

    Raises:
        ValueError: If input data shape is invalid or vocab sizes mismatch config.
    """
    b, l = peaks.shape
    if l % 3 != 0:
        raise ValueError(f"Peak tensor length {l} must be divisible by 3 (for c13_centroid, h1_centroid, nh)")

    seq_len = l // 3
    peaks = peaks.to(device).reshape(b, 3, seq_len)  # Shape: (batch_size, 3, sequence_length)

    c13_centroid = peaks[:, 0, :]  # 13C_centroid
    h1_centroid = peaks[:, 1, :]  # 1H_centroid
    nh = peaks[:, 2, :]  # nH

    hsqc_idx = [c13_centroid, h1_centroid, nh]

    # Create mask for non-special tokens (excluding BOS, EOS, PAD)
    non_special_mask_c13 = torch.logical_and(
        torch.logical_and(c13_centroid != test_set.CONFIG['nmr_bos_token'],
                          c13_centroid != test_set.CONFIG['nmr_eos_token']),
        c13_centroid != test_set.CONFIG['nmr_pad_token']
    )
    non_special_mask_h1 = torch.logical_and(
        torch.logical_and(h1_centroid != test_set.CONFIG['nmr_bos_token'],
                          h1_centroid != test_set.CONFIG['nmr_eos_token']),
        h1_centroid != test_set.CONFIG['nmr_pad_token']
    )
    # nH is typically not masked, but include for consistency
    non_special_mask_nh = torch.logical_and(
        torch.logical_and(nh != test_set.CONFIG['nmr_bos_token'],
                          nh != test_set.CONFIG['nmr_eos_token']),
        nh != test_set.CONFIG['nmr_pad_token']
    )

    # Apply noise to centroids
    if noise_range > 0:
        centroids_noise = torch.randint(low=-noise_range, high=noise_range + 1, size=c13_centroid.shape, device=device)

        # Apply noise to c13_centroid, clamp to valid range
        c13_centroid_pertub = torch.where(non_special_mask_c13,
                                          torch.clamp(c13_centroid + centroids_noise,
                                                      min=test_set.CONFIG['nmr_pad_token'] + 1,
                                                      max=test_set.CONFIG['c_nmr_delta_disc']),
                                          c13_centroid)

        # Apply noise to h1_centroid, clamp to valid range
        h1_centroid_pertub = torch.where(non_special_mask_h1,
                                         torch.clamp(h1_centroid + centroids_noise,
                                                     min=test_set.CONFIG['nmr_pad_token'] + 1,
                                                     max=test_set.CONFIG['centroid_disc']),
                                         h1_centroid)

        # nH typically not noised, but include for flexibility
        nh_pertub = torch.where(non_special_mask_nh,
                                torch.clamp(nh + centroids_noise,
                                            min=test_set.CONFIG['nmr_pad_token'] + 1,
                                            max=test_set.CONFIG['max_nH']),
                                nh)

        hsqc_idx_pertub = [c13_centroid_pertub, h1_centroid_pertub, nh_pertub]
    else:
        hsqc_idx_pertub = hsqc_idx

    cond = None  # Replace with dataset-provided condition if available
    return hsqc_idx_pertub, cond
