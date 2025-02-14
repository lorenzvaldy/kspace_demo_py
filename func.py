from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tkinter import ttk, StringVar, BooleanVar, filedialog
from typing import Dict, Any, Tuple
from scipy.interpolate import griddata
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from pynput import keyboard

def kspace_params_gui(params: Dict[str, Any] = None) -> Tuple[Dict[str, Any], bool]:
    """
    Create a unified GUI for setting parameters for the k-space demo.
    Returns parameters dictionary and OK/cancel status.
    """
    # Initialize parameters if not provided
    params = params.copy() if params else {}

    # Create main window
    root = tk.Tk()
    root.title("K-Space Parameters Configuration")
    
    # Create main container
    main_frame = ttk.Frame(root, padding="20 10 20 20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Field configuration
    fields = [
        ("imfile", "Image file:", "popup", ["imgs/axialBrain.jpg", "other"], params.get("imfile", "imgs/axialBrain.jpg")),
        ("sequenceType", "Acquisition Sequence:", "popup", ["spiral", "EPI"], params.get("sequenceType", "EPI")),
        ("noiseType", "B0 noise type:", "popup", ["local offset", "random offset", "random lowpass", 
                "x gradient", "y gradient", "dc offset", "map", "none"], params.get("noiseType", "local offset")),
        ("noiseScale", "Noise scale (ppm):", "number", params.get("noiseScale", 0.5) * 1e6),
        ("FOV", "Field of View (mm):", "number", params.get("FOV", 180) * 1e3),
        ("res", "Recon pixel size (mm):", "number", params.get("res", 2) * 1e3),
        ("imSize", "Original image size (mm):", "number", params.get("imSize", 180) * 1e3),
        ("imRes", "Original pixel size (mm):", "number", params.get("imRes", 1) * 1e3),
        ("bandwidth", "Bandwidth (KHz):", "number", params.get("bandwidth", 125) / 1e3),
        ("echoTime", "Echo time (ms):", "number", params.get("echoTime", 30) * 1e3),
        ("oversample", "Oversample (n shots):", "number", params.get("oversample", 1)),
        ("showProgress", "Show progressive recon", "checkbox", params.get("showProgress", False)),
        ("loop", "Keep GUI open", "checkbox", params.get("loop", True)),
    ]

    # Storage for widget variables
    widgets = {}
    row = 0

    # Create input widgets
    for field in fields:
        field_name, label_text, field_type, *args = field
        var = None
        
        ttk.Label(main_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            
        if field_type == "popup":
            options, default = args[0], args[1]
            var = StringVar(value=default)
            if field_name == "imfile":
                prev_value = [default]  # Store previous value in a mutable container
                
                def file_browser_callback(*args, var=var, prev=prev_value):
                    current = var.get()
                    if current == "other":
                        filepath = filedialog.askopenfilename(
                            title="Select Image File",
                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
                        )
                        if filepath:
                            var.set(filepath)
                            prev[0] = filepath
                        else:
                            var.set(prev[0])  # Revert to previous value if canceled
                    else:
                        prev[0] = current  # Update previous value
                    
                var.trace_add("write", file_browser_callback)
                
            widget = ttk.OptionMenu(main_frame, var, default, *options)
            widget.grid(row=row, column=1, sticky=tk.EW, padx=5)
            
        elif field_type == "checkbox":
            default = args[0]
            var = BooleanVar(value=default)
            widget = ttk.Checkbutton(main_frame, variable=var)
            widget.grid(row=row, column=1, sticky=tk.W, padx=5)
            
        elif field_type == "number":
            default = args[0]
            var = tk.DoubleVar(value=default)
            widget = ttk.Entry(main_frame, textvariable=var, width=10)
            widget.grid(row=row, column=1, sticky=tk.EW, padx=5)
            
        widgets[field_name] = var
        row += 1

    # Add buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=row+1, column=0, columnspan=2, pady=10)
    
    ok = False

    def on_ok():
        nonlocal ok
        ok = True
        root.destroy()

    def on_cancel():
        root.destroy()

    ttk.Button(button_frame, text="OK", command=on_ok).grid(row=0, column=0, padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=5)

    # Center the window and run
    root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))
    root.mainloop()

    if not ok:
        return params, False

    # Collect all values
    user_inputs = {}
    for field_name, var in widgets.items():
        user_inputs[field_name] = var.get()

    # Update parameters with user inputs
    params.update(user_inputs)

    # Convert units and derive parameters (same as original)
    params["gamma"] = 42.58e6 * 2 * np.pi
    params["B0"] = 3.0

    # Convert millimeters to meters
    for key in ["res", "FOV", "imSize", "imRes"]:
        params[key] = params[key] / 1e3

    params["FOV"] = round(params["FOV"] / params["res"] / 2) * params["res"] * 2
    params["freq"] = int(params["FOV"] / params["res"])
    params["imFreq"] = int(params["imSize"] / params["imRes"])

    # Convert kilohertz to hertz
    params["bandwidth"] = params["bandwidth"] * 1e3
    params["dt"] = 1 / params["bandwidth"]
    params["echoTime"] = params["echoTime"] / 1e3
    params["noiseScale"] = params["noiseScale"] * 1e-6

    # Gradient strength
    params["gx"] = 2 * np.pi / (params["gamma"] * params["dt"])
    params["gy"] = params["gx"]

    return params, True

def kspace_get_image(k):
    # Import necessary libraries
    import numpy as np
    from PIL import Image
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    # Initialize the output dictionary
    im = {}

    imfile = k['imfile']
    sz = int(k['imFreq']) # Number of pixels along one side of the image (assumed to be square)

    # If not a default image, let user select it from a prompt
    if imfile.lower() == 'other':
        fname = ''
        while not fname:
            # Hide the root window of Tkinter
            root = Tk()
            root.withdraw()
            fname = askopenfilename(title='Pick any image', filetypes=[('All files', '*.*')])
            root.destroy()
        imfile = fname

    # Load the image
    im_orig = Image.open(imfile)

    # Ensure the image is in grayscale
    if im_orig.mode != 'L':
        im_orig = im_orig.convert('L')

    # Resize the image to the desired resolution
    im_orig = im_orig.resize((sz, sz))

    # Store the original image
    im['orig'] = im_orig

    # Convert the image to a NumPy array for processing
    im_array = np.array(im_orig)

    # Flatten the image to a vector and convert to float for calculations
    im['vector'] = im_array.flatten().astype(np.float64)

    # Compute the 2D FFT of the image
    im['fft'] = np.fft.fft2(im_array)

    # Compute the shifted FFT for visualization purposes
    im['fftshift'] = np.fft.fftshift(np.log(np.abs(im['fft'])))

    return im

def kspace_grid(k):
    """
    Returns x, y (meters) grid representing voxel positions.

    Parameters:
        k (dict): A dictionary containing imaging parameters with keys:
            - 'imSize': The width of the image in meters.
            - 'imFreq': The number of pixels along one side of the image.
                        (Should satisfy imSize = imFreq * imRes)
            - 'imRes': The width (and height) of one pixel in meters.

    Returns:
        dict: A dictionary with keys 'x' and 'y' containing 2D NumPy arrays
            representing the grid of voxel positions.
    """
    # Extract parameters from the input dictionary 'k'
    sz = k['imSize']    # Width of the image in meters
    xfreq = k['imFreq'] # Number of pixels along the x-axis
    yfreq = k['imFreq'] # Number of pixels along the y-axis (assuming square pixels)

    # Calculate the width and height of one pixel in meters
    dx = sz / xfreq     # Width of one pixel
    dy = sz / yfreq     # Height of one pixel

    # Generate linearly spaced coordinates along x and y axes
    # The coordinates range from 0 to sz - dx/dy, with a total of xfreq/yfreq points
    x_axis = np.linspace(0, sz - dx, xfreq)
    y_axis = np.linspace(0, sz - dy, yfreq)

    # Create 2D grids for x and y coordinates
    x, y = np.meshgrid(x_axis, y_axis)

    # Store the grids in a dictionary and return
    xygrid = {'x': x, 'y': y}

    return xygrid

def kspace_prepare_figure():
    """
    Prepare a figure window for the k-space demo. 
    The figure uses the full height of the display and half the width.
    The figure is set to figure(1). Returns the figure object.

    Returns:
        fig (matplotlib.figure.Figure): The prepared figure object.
    """
    # Create a new figure
    fig = plt.figure(1)
    
    # Get the screen size
    screen_width, screen_height = plt.gcf().canvas.get_width_height()
    
    # Set the figure size to full height and half width
    fig.set_size_inches(screen_height / 50, screen_width / 140)  # Adjust scaling as needed
    
    # Set the figure position (optional in Python, as window management is handled by the OS)
    # For full control, you can use a GUI toolkit like Tkinter or PyQt.
    
    return fig

def kspace_epi(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Generate EPI gradient sequence.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters, including:
            - freq (int): Frequency resolution (number of samples per row/column).
            - FOV (float): Field of view (used to calculate kx).

    Returns:
        gradients (Dict[str, np.ndarray]): A dictionary containing:
            - 'x': Phase encode gradient (x-direction).
            - 'y': Frequency encode gradient (y-direction).
            - 'T': Time vector (dummy values for now).
    """
    # Initialize variables
    freq = int(params['freq'])
    nsamples = int(freq**2)  # Number of points to acquire (square of resolution)
    kx = 1 / params['FOV']  # Scaling factor for gradients

    # Initialize gradients
    gradients = {
        'T': np.ones(nsamples),  # Time vector (dummy values)
        'x': np.zeros(nsamples),  # Phase encode gradient (x-direction)
        'y': np.zeros(nsamples)   # Frequency encode gradient (y-direction)
    }

    # Frequency encode (y gradient)
    for row in range(1, freq + 1, 2):
        inds = np.arange((row - 1) * freq, row * freq)
        gradients['y'][inds] = kx  # Positive gradient for odd rows
        if row + 1 <= freq:
            gradients['y'][inds + freq] = -kx  # Negative gradient for even rows

    # Phase encode (x gradient)
    phase_blip_inds = np.arange(freq, nsamples - freq + 1, freq)
    gradients['x'][phase_blip_inds] = kx  # Phase encode blips
    gradients['y'][phase_blip_inds] = 0   # Shut off frequency gradient during phase blips

    # Initial point (move to upper left of k-space)
    gradients['x'][0] = -kx
    gradients['y'][0] = -kx
    gradients['T'][0] = freq / 2  # Adjust initial time step
    
    return gradients

def kspace_spiral(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Generate a spiral-out gradient sequence.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters, including:
            - freq (int): Frequency resolution (number of samples per row/column).
            - oversample (int): Number of extra spiral shots (oversampling).
            - FOV (float): Field of view (used to calculate gradients).

    Returns:
        gradients (Dict[str, np.ndarray]): A dictionary containing:
            - 'x': Phase encode gradient (x-direction).
            - 'y': Frequency encode gradient (y-direction).
            - 'T': Time vector (dummy values for now).
    """
    # Initialize variables
    nsamples = int(params['freq'] ** 2)  # Number of k-space samples
    noversample = int(params['oversample'])  # Number of oversampling shots
    FOV = params['FOV']  # Field of view

    # Initialize gradients
    gradients = {
        'T': np.ones(nsamples * noversample),  # Time vector (dummy values)
        'x': np.zeros(nsamples * noversample),  # Phase encode gradient (x-direction)
        'y': np.zeros(nsamples * noversample)   # Frequency encode gradient (y-direction)
    }

    for ii in range(noversample):
        # Define theta (angle for spiral trajectory)
        theta = np.linspace(0, params['freq'] * np.pi, nsamples)
        dtheta = np.diff(theta).mean()  # Angular step size

        # Calculate indices for this interleave
        inds = np.arange(nsamples) + ii * nsamples

        # Angular offset for interleaves
        offset = ii / noversample * 2 * np.pi

        # Calculate gradients (x and y)
        gradients['x'][inds] = (dtheta / (2 * np.pi * FOV)) * (
            np.sin(theta + offset) + theta * np.cos(theta + offset)
        )
        gradients['y'][inds] = (dtheta / (2 * np.pi * FOV)) * (
            np.cos(theta + offset) + theta * np.sin(theta + offset)
        )

        # Return to the center of k-space if doing multiple interleaves
        if ii > 0:
            previous_inds = np.arange(inds[0])  # Indices of previous interleaves
            xpos = np.sum(gradients['x'][previous_inds] * gradients['T'][previous_inds])
            ypos = np.sum(gradients['y'][previous_inds] * gradients['T'][previous_inds])
            gradients['x'][inds[0]] = -xpos / len(previous_inds)
            gradients['y'][inds[0]] = -ypos / len(previous_inds)
            gradients['T'][inds[0]] = len(previous_inds)

    return gradients

def kspace_initialize_matrices(params: Dict[str, Any], gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Initialize k-space matrices.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters, including:
            - freq (int): Frequency resolution (number of samples per row/column).
            - res (float): Resolution (used to calculate spatial frequencies).
        gradients (Dict[str, np.ndarray]): A dictionary containing:
            - 'x': Phase encode gradient (x-direction).
            - 'y': Frequency encode gradient (y-direction).
            - 'T': Time vector.

    Returns:
        kspace (Dict[str, Any]): A dictionary containing:
            - 'vector': A dictionary with k-space vector data (x, y, real, imag).
            - 'grid': A dictionary with k-space grid data (x, y, real, imag).
    """
    # Initialize k-space structure
    kspace = {
        'vector': {},
        'grid': {}
    }

    # Vectors
    # Convert gradient values to k-space positions
    # Position in k-space is the time integral of the gradients
    T = gradients['T']
    kspace['vector']['x'] = np.cumsum(gradients['x'] * gradients['T'])
    kspace['vector']['y'] = np.cumsum(gradients['y'] * gradients['T'])

    # Initialize real and imaginary parts of the k-space vector
    kspace['vector']['real'] = np.zeros_like(T)
    kspace['vector']['imag'] = np.zeros_like(T)

    # Grids
    # Define the xy indices by the spatial frequencies
    nsamples = params['freq']
    freqs = np.linspace(-0.5, 0.5, nsamples + 1) * (1 / params['res'])
    freqs = freqs[:-1]  # Remove the last frequency to match MATLAB's behavior

    # Create a meshgrid for the k-space grid
    kspace['grid']['x'], kspace['grid']['y'] = np.meshgrid(freqs, freqs)

    # Shift the grid to match MATLAB's fftshift behavior
    kspace['grid']['x'] = np.fft.fftshift(kspace['grid']['x'])
    kspace['grid']['y'] = np.fft.fftshift(kspace['grid']['y'])

    # Initialize empty real and imaginary parts of the k-space grid
    kspace['grid']['real'] = np.zeros((nsamples, nsamples))
    kspace['grid']['imag'] = np.zeros((nsamples, nsamples))

    return kspace

def kspace_precompute(params: Dict[str, Any], gradients: Dict[str, np.ndarray],
                     xygrid: Dict[str, np.ndarray], b0noise: np.ndarray) -> Dict[str, Any]:
    """
    Precompute spin phases for unique gradient combinations to optimize computation.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters.
        gradients (Dict[str, np.ndarray]): A dictionary containing gradient sequences:
            - 'x': Phase encode gradient (x-direction).
            - 'y': Frequency encode gradient (y-direction).
            - 'T': Time vector.
        xygrid (Dict[str, np.ndarray]): A dictionary containing x and y grid coordinates.
        b0noise (np.ndarray): B0 inhomogeneity map.

    Returns:
        spins (Dict[str, Any]): A dictionary containing precomputed spin phases and indices.
    """
    # Initialize spins dictionary
    spins = {}

    # Calculate the effect of the echo time
    spins = kspace_compute_one_point(params, gradients, xygrid, b0noise, spins, 'start')

    # Update the spin phase to account for the most recent step
    spins = kspace_get_current_basis_functions(spins)

    # Find unique gradient combinations
    gradient_combinations = np.column_stack([gradients['y'], gradients['x'], gradients['T']])
    unique_combinations, unique_indices, inverse_indices = np.unique(
        gradient_combinations, axis=0, return_index=True, return_inverse=True
    )

    # If there are too many unique values, skip precomputing
    if len(unique_combinations) > 20:
        return spins

    # Precompute spin phases for unique gradient combinations
    spins['precompute'] = np.zeros(
        (xygrid['x'].shape[0], xygrid['y'].shape[1], len(unique_combinations)), dtype=np.complex64
    )
    for ii in range(len(unique_combinations)):
        t = unique_indices[ii]
        tmpspins = kspace_compute_one_point(params, gradients, xygrid, b0noise, {}, t)
        spins['precompute'][:, :, ii] = tmpspins['step']

    # Store the indices for mapping back to the full gradient sequence
    spins['precompute_index'] = inverse_indices

    return spins

def kspace_get_current_signal(kspace: Dict[str, Any], t: int, im: Dict[str, np.ndarray],
                             spins: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the k-space signal for the current time point.

    Args:
        kspace (Dict[str, Any]): A dictionary containing k-space data:
            - 'vector': A dictionary with 'real' and 'imag' arrays.
        t (int): Time point index.
        im (Dict[str, np.ndarray]): A dictionary containing the image vector.
        spins (Dict[str, Any]): A dictionary containing spin phases:
            - 'total': The cumulative spin phase (complex array).
        params (Dict[str, Any]): A dictionary containing parameters:
            - 'imRes': Image resolution (pixel size in meters).

    Returns:
        kspace (Dict[str, Any]): Updated k-space dictionary with the new signal values.
    """
    # Extract parameters
    dx = params['imRes']  # Pixel size in x-direction (meters)
    dy = params['imRes']  # Pixel size in y-direction (meters)

    # Extract real and imaginary parts of the cumulative spin phase
    s_r = np.real(spins['total'])  # Real (sinusoidal) basis matrix
    s_i = np.imag(spins['total'])  # Imaginary (cosinusoidal) basis matrix

    # Extract the image vector
    imv = im['vector']

    # Compute the k-space signal for the current time point
    kspace['vector']['real'][t] = np.dot(imv, s_r.flatten()) * dx * dy
    kspace['vector']['imag'][t] = np.dot(imv, s_i.flatten()) * dx * dy

    return kspace

def kspace_recon(kspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct k-space data into a grid using interpolation or Voronoi weighting.

    Args:
        kspace (Dict[str, Any]): A dictionary containing k-space data:
            - 'vector': A dictionary with 'x', 'y', 'real', and 'imag' arrays.
            - 'grid': A dictionary with 'x' and 'y' grid coordinates.
        params (Dict[str, Any]): A dictionary containing parameters:
            - 'sequenceType': Type of sequence ('epi' or 'spiral').
            - 'freq': Frequency resolution (number of samples per row/column).
            - 'FOV': Field of view (used to calculate kmax).

    Returns:
        kspace (Dict[str, Any]): Updated k-space dictionary with reconstructed grid data.
    """
    sequence_type = params['sequenceType'].lower()
    v = kspace['vector']
    g = kspace['grid']

    method = 'griddata'  # Default method

    if sequence_type == 'epi':
        # Use griddata interpolation for EPI sequences
        kr = griddata((v['x'], v['y']), v['real'], (g['x'], g['y']), method='cubic')
        ki = griddata((v['x'], v['y']), v['imag'], (g['x'], g['y']), method='cubic')

    elif sequence_type == 'spiral':
        if method == 'griddata':
            # Ignore measurements beyond our resolution (k > kmax)
            kmax = params['freq'] / params['FOV'] * 0.5  # cycles / meter
            inds = (np.abs(v['x']) <= kmax) & (np.abs(v['y']) <= kmax)

            # Reconstruct using griddata interpolation
            kr = griddata((v['x'][inds], v['y'][inds]), v['real'][inds], (g['x'], g['y']), method='cubic')
            ki = griddata((v['x'][inds], v['y'][inds]), v['imag'][inds], (g['x'], g['y']), method='cubic')

        elif method == 'voronoi':
            # Reconstruct using Voronoi weighting (placeholder implementation)
            # This part requires a custom implementation of the Voronoi weighting function.
            # For now, we use griddata as a placeholder.
            kr = griddata((v['x'], v['y']), v['real'], (g['x'], g['y']), method='cubic')
            ki = griddata((v['x'], v['y']), v['imag'], (g['x'], g['y']), method='cubic')
            print("Voronoi weighting is not implemented. Using griddata instead.")

    # Handle NaNs and very small values
    kr[np.isnan(kr) | (np.abs(kr) < 1e-15)] = 0
    ki[np.isnan(ki) | (np.abs(ki) < 1e-15)] = 0

    # Update the k-space grid with reconstructed data
    kspace['grid']['real'] = kr
    kspace['grid']['imag'] = ki

    return kspace

def kspace_get_b0_noise(params: Dict[str, Any], xygrid: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a map of B0 field inhomogeneities in units of Tesla.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters:
            - 'B0': Main magnetic field strength (Tesla).
            - 'imFreq': Image frequency resolution (number of samples per row/column).
            - 'noiseType': Type of noise ('none', 'random', 'random lowpass', 'dc offset',
                             'x gradient', 'y gradient', 'local', 'map').
            - 'noiseScale': Scaling factor for the noise amplitude.
            - 'imSize': Image size (meters).
            - 'gamma': Gyromagnetic ratio (Hz/Tesla).
        xygrid (Dict[str, np.ndarray]): A dictionary containing x and y grid coordinates.

    Returns:
        b0noise (np.ndarray): A 2D array representing the B0 inhomogeneity map.
    """
    # Extract parameters
    B0 = params['B0']
    freq = params['imFreq']
    noise_type = params['noiseType'].lower()
    noise_scale = params['noiseScale']
    sz = params['imSize']
    x = xygrid['x']
    y = xygrid['y']

    # Make the spatial pattern of B0 noise, assuming an amplitude of 1
    if noise_type == 'none':
        b0noise = np.zeros((freq, freq))
    elif noise_type in ['random', 'random offset']:
        b0noise = np.random.randn(freq, freq)
        max_val = np.max(np.abs(b0noise))
        if max_val != 0:  # Avoid division by zero
            b0noise /= max_val
    elif noise_type == 'random lowpass':
        # Blobby, low-pass noise
        b0noise = np.random.randn(freq, freq)
        sigma = freq / 16  # Standard deviation for the Gaussian filter
        b0noise = gaussian_filter(b0noise, sigma=sigma, mode='constant')
        max_val = np.max(np.abs(b0noise))
        if max_val != 0:  # Avoid division by zero
            b0noise /= max_val
    elif noise_type in ['dc offset', 'constant', 'uniform']:
        # Uniform DC offset
        b0noise = np.ones((freq, freq))
    elif noise_type == 'x gradient':
        # Spatially varying in the x-direction
        b0noise = x / np.max(np.abs(x))
        b0noise -= np.mean(b0noise)
    elif noise_type == 'y gradient':
        # Spatially varying in the y-direction
        b0noise = y / np.max(np.abs(y))
        b0noise -= np.mean(b0noise)
    elif noise_type in ['local', 'local offset']:
        # Define noise center and radius
        noise_center = [0.25 * sz, 0.25 * sz]  # Center of the noise spot
        noise_radius = 0.01 * sz  # Radius of the noise spot

        # Create a grid of x and y coordinates
        x = np.linspace(0, sz, freq)
        y = np.linspace(0, sz, freq)
        x, y = np.meshgrid(x, y)

        # Create a binary mask for the noise spot
        dist = np.sqrt((x - noise_center[0]) ** 2 + (y - noise_center[1]) ** 2)
        inds = dist < noise_radius
        b0noise = np.zeros((freq, freq))
        b0noise[inds] = 1

        # Apply a Gaussian filter
        sigma = freq / 16  # Standard deviation for the Gaussian filter
        b0noise = gaussian_filter(b0noise, sigma=sigma, mode='constant')

        # Normalize the noise map
        max_val = np.max(np.abs(b0noise))
        if max_val != 0:  # Avoid division by zero
            b0noise /= max_val
    elif noise_type == 'map':
        # Load a precomputed B0 map (placeholder implementation)
        try:
            b0noise = np.load('b0Lucas.npy')  # Load B0 map in Hz
            b0noise = b0noise * (2 * np.pi / params['gamma'])  # Convert from Hz to Tesla
            b0noise = np.resize(b0noise, (freq, freq))
            return b0noise  # Skip scaling step for real quantitative B0 map
        except FileNotFoundError:
            print("B0 map file 'b0Lucas.npy' not found. Using zeros instead.")
            b0noise = np.zeros((freq, freq))
    else:
        b0noise = np.zeros((freq, freq))

    # Scale the amplitude of the B0 noise image as requested
    b0noise *= noise_scale * B0

    return b0noise

def kspace_make_pulse_sequence(params: Dict[str, Any]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Get gradients and expected k-space locations as a function of time.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters, including:
            - sequenceType (str): Type of sequence ('epi' or 'spiral').

    Returns:
        gradients (Dict[str, np.ndarray]): Gradient sequence.
        kspace (Dict[str, np.ndarray]): K-space locations through the sequence.
    """
    sequence_type = params['sequenceType'].lower()

    if sequence_type == 'spiral':
        gradients = kspace_spiral(params)
    elif sequence_type == 'epi':
        gradients = kspace_epi(params)
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    # Initialize k-space matrices
    kspace = kspace_initialize_matrices(params, gradients)

    return gradients, kspace

def kspace_compute_one_point(params: Dict[str, Any], gradients: Dict[str, np.ndarray],
                            xygrid: Dict[str, np.ndarray], b0noise: np.ndarray,
                            spins: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Calculate the spin phase at each discrete image location from the gradients and B0 inhomogeneities.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters:
            - gamma (float): Gyromagnetic constant for hydrogen.
            - dt (float): Time for discrete step in k-space sampling.
            - gx (float): Gradient constant in x-direction (Tesla/meter).
            - gy (float): Gradient constant in y-direction (Tesla/meter).
            - echoTime (float): Echo time.
            - noiseType (str): Type of noise ('none' or other).
        gradients (Dict[str, np.ndarray]): A dictionary containing gradient sequences:
            - 'x': Phase encode gradient (x-direction).
            - 'y': Frequency encode gradient (y-direction).
            - 'T': Time vector.
        xygrid (Dict[str, np.ndarray]): A dictionary containing x and y grid coordinates.
        b0noise (np.ndarray): B0 inhomogeneity map.
        spins (Dict[str, Any]): A dictionary containing spin phases.
        t (int): Time point index.

    Returns:
        spins (Dict[str, Any]): Updated spins dictionary with the new spin phase step.
    """
    # Check if precomputed spin states are available
    if 'precompute' in spins:
        spins['step'] = spins['precompute'][:, :, spins['precompute_index'][t]]
        return spins

    # Constants
    gamma = params['gamma']  # Gyromagnetic constant for hydrogen
    dt = params['dt']        # Time step for k-space sampling
    gx = params['gx']        # Gradient constant in x-direction (Tesla/meter)
    gy = params['gy']        # Gradient constant in y-direction (Tesla/meter)

    # Spatial locations in image (in meters)
    x = xygrid['x']
    y = xygrid['y']

    # Gradients (these change over time)
    if t == 'start':
        # Effect of waiting TE (echo time): only B0 noise contributes
        GX = 0
        GY = 0
        T = params['echoTime'] / dt
    else:
        # Gradients are active
        GX = gradients['x'][t]
        GY = gradients['y'][t]
        T = gradients['T'][t]

    # Compute spin change due to x-gradient
    step_x = np.exp(-1j * gamma * x * gx * GX * dt * T)
    
    # Compute spin change due to y-gradient
    step_y = np.exp(-1j * gamma * y * gy * GY * dt * T)

    # Compute total spin change
    if params['noiseType'].lower() == 'none':
        step = step_x * step_y  # No B0 noise
    else:
        # Compute spin change due to B0 noise
        step_e = np.exp(-1j * T * dt * b0noise * gamma)
        step = step_x * step_y * step_e

    # Store the computed step in the spins dictionary
    spins['step'] = step

    return spins

def kspace_get_current_basis_functions(spins: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the spins to account for the most recent step.

    Args:
        spins (Dict[str, Any]): A dictionary containing spin phases:
            - 'step': The most recent spin phase step.
            - 'total': The cumulative spin phase (optional).

    Returns:
        spins (Dict[str, Any]): Updated spins dictionary with the cumulative spin phase.
    """
    if 'total' not in spins:
        # If this is the first data acquisition
        spins['total'] = spins['step']
    else:
        # Update the cumulative spin phase
        spins['total'] = spins['total'] * spins['step']

    return spins

def kspace_show_plots(f: plt.Figure, spins: Dict[str, Any], gradients: Dict[str, np.ndarray],
                     kspace: Dict[str, Any], im: Dict[str, np.ndarray], params: Dict[str, Any],
                     t: int, b0noise: np.ndarray, M: list) -> tuple[Dict[str, Any], list]:
    """
    Visualize k-space data, reconstructed images, gradients, and B0 maps.

    Args:
        f (plt.Figure): The figure object for plotting.
        spins (Dict[str, Any]): A dictionary containing spin phases.
        gradients (Dict[str, np.ndarray]): A dictionary containing gradient sequences.
        kspace (Dict[str, Any]): A dictionary containing k-space data.
        im (Dict[str, np.ndarray]): A dictionary containing the original image and its FFT.
        params (Dict[str, Any]): A dictionary containing parameters.
        t (int): The current time point index.
        b0noise (np.ndarray): The B0 inhomogeneity map.
        M (list): A list to store frames for creating a movie.

    Returns:
        kspace (Dict[str, Any]): Updated k-space dictionary.
        M (list): Updated list of frames for creating a movie.
    """
    # If not done filling k-space and not showing step-by-step progress, return
    #if (t < len(kspace['vector']['x']) and not params['showProgress']) and not (t == len(kspace['vector']['x']) - 1):
    if (t < len(kspace['vector']['x']) and not params['showProgress']) and not (t == len(kspace['vector']['x']) - 1):
        return kspace, M

    # Reconstruct k-space
    kspace = kspace_recon(kspace, params)

    # Set up plots and subplots
    rows, cols = 3, 2
    n = 1

    # Check if this is the first time plotting
    if hasattr(f, 'userData') and t > 1:
        initialize = False
        subplot_handle = f.userData['subplot_handle']
    else:
        initialize = True
        f.userData = {'initialized': True, 'subplot_handle': {}}
        
    if initialize:
        if plt.fignum_exists(f.number):  # Check if the figure already exists
            plt.close(f.number)  # Close the existing figure
        # Create a new figure with the desired size
        plt.figure(f.number, figsize=(16, 12))
        plt.clf()
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Add more space between subplots
        
    # Plot 1: k-space filled by imaging
    tmp = np.fft.fftshift(np.log(np.abs(kspace['grid']['real'] + 1j * kspace['grid']['imag']) + 1e-10))

    x = np.fft.fftshift(kspace['grid']['x'])
    y = np.fft.fftshift(kspace['grid']['y'])
    ma = np.max(tmp[np.isfinite(tmp)], initial=0)
    mi = np.min(tmp[np.isfinite(tmp)], initial=0)
    if ma <= mi:
        ma, mi = 1, 0
    if initialize:
        plt.subplot(rows, cols, n)
        plt.imshow(tmp, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray', vmin=mi, vmax=ma)
        plt.title('k-space filled by imaging')
        plt.xlabel('cycles per meter')
        plt.ylabel('cycles per meter')
        plt.axis('image')
        f.userData['subplot_handle'][n] = plt.gca()
    else:
        plt.sca(f.userData['subplot_handle'][n])
        plt.imshow(tmp, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray', vmin=mi, vmax=ma)
    n += 1

    # Plot 2: Image reconstructed from k-space
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace['grid']['real'] + 1j * kspace['grid']['imag'])))
    imsize = [0, params['imSize'] * 100]
    if initialize:
        plt.subplot(rows, cols, n)
        plt.imshow(recon, extent=imsize + imsize, cmap='gray')
        plt.title('Image reconstructed from k-space')
        plt.xlabel('mm')
        plt.ylabel('mm')
        ticks = np.linspace(0, max(imsize), 5)
        plt.xticks(ticks)
        plt.yticks(ticks)
        f.userData['subplot_handle'][n] = plt.gca()
    else:
        plt.sca(f.userData['subplot_handle'][n])
        plt.imshow(recon, extent=imsize + imsize, cmap='gray')
    n += 1

    # Plot 3: k-space computed from image
    if initialize:
        plt.subplot(rows, cols, n)
        plt.imshow((im['fftshift']), extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
        plt.title('k-space computed from image')
        plt.xlabel('cycles per meter')
        plt.ylabel('cycles per meter')
        plt.axis('image')
        f.userData['subplot_handle'][n] = plt.gca()
    n += 1

    # Plot 4: Original Image
    if initialize:
        plt.subplot(rows, cols, n)
        plt.imshow(im['orig'], extent=imsize + imsize, cmap='gray')
        plt.title('Original Image')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.xticks(ticks)
        plt.yticks(ticks)
        f.userData['subplot_handle'][n] = plt.gca()
    n += 1

    # Plot 5: Sinusoidal spin channel
    if initialize:
        plt.subplot(rows, cols, n)
        plt.imshow(np.real(spins['total']), cmap='gray')
        plt.title(f'REAL: x={kspace["vector"]["x"][t]:.1f} cpm, y={kspace["vector"]["y"][t]:.1f} cpm')
        plt.axis('off')
        f.userData['subplot_handle'][n] = plt.gca()
    else:
        plt.sca(f.userData['subplot_handle'][n])
        plt.imshow(np.real(spins['total']), cmap='gray')
    n += 1

    # Plot 6: B0 Map
    if initialize:
        plt.subplot(rows, cols, n)
        rg = [-1 * np.max(np.abs(b0noise)), 1 * np.max(np.abs(b0noise))]
        if np.all(rg == 0):
            rg = [-1 * np.finfo(float).eps, 1 * np.finfo(float).eps]
        plt.imshow(b0noise, extent=imsize + imsize, cmap='gray', vmin=rg[0], vmax=rg[1])
        noise_min = rg[0] * params['gamma'] / (2 * np.pi)
        noise_max = rg[1] * params['gamma'] / (2 * np.pi)
        plt.title(f'B0 map. Range = [{noise_min:.1f} {noise_max:.1f}] Hz')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.grid(True, color='white', linestyle='-')
        plt.xticks(ticks)
        plt.yticks(ticks)
        f.userData['subplot_handle'][n] = plt.gca()

    # Draw the figure
    plt.draw()
    plt.pause(0.01)

    # Capture frame for movie
    if params['showProgress']:
        M.append(plt.gcf())

    # Plot gradients in a separate figure
    plt.figure(f.number + 1)
    if initialize:
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot([0, *np.cumsum(gradients['T'][:t])], [gradients['y'][0], *gradients['y'][:t]], 'g-', linewidth=1)
        plt.title('Gradients')
        plt.ylim([-1.1 * np.max(gradients['x']), 1.1 * np.max(gradients['x'])])
        plt.xlim([0, np.sum(gradients['T'])])
        f.userData['subplot_handle']['grad_y'] = plt.gca()

        plt.subplot(2, 1, 2)
        plt.plot([0, *np.cumsum(gradients['T'][:t])], [gradients['x'][0], *gradients['x'][:t]], 'r-', linewidth=1)
        plt.ylim([-1.1 * np.max(gradients['x']), 1.1 * np.max(gradients['x'])])
        plt.xlim([0, np.sum(gradients['T'])])
        f.userData['subplot_handle']['grad_x'] = plt.gca()
    else:
        plt.sca(f.userData['subplot_handle']['grad_y'])
        plt.plot([0, *np.cumsum(gradients['T'][:t])], [gradients['y'][0], *gradients['y'][:t]], 'g-', linewidth=1)
        plt.sca(f.userData['subplot_handle']['grad_x'])
        plt.plot([0, *np.cumsum(gradients['T'][:t])], [gradients['x'][0], *gradients['x'][:t]], 'r-', linewidth=1)
        
    # Draw the figure
    plt.draw()
    
    if not params['showProgress']:
        plt.show(block=True)  # Wait until figure is closed
    
    return kspace, M

def kspace_demo(params: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    A demonstration of MRI imaging principles and artifacts.

    Args:
        params (Dict[str, Any]): A dictionary containing parameters (optional).

    Returns:
        params (Dict[str, Any]): Updated parameters.
        im (Dict[str, Any]): Image data.
        kspace (Dict[str, Any]): K-space data.
        spins (Dict[str, Any]): Spin data.
        gradients (Dict[str, Any]): Gradient data.
    """
    # Check inputs
    if params is None:
        params = {}

    # Define scan parameters and constants
    params, ok = kspace_params_gui(params)
    if not ok:
        return params, {}, {}, {}, {}

    # Read in an image
    im = kspace_get_image(params)

    # Create a grid of x, y values (in meters) to represent pixel positions
    xygrid = kspace_grid(params)

    # Create a map of B0 field inhomogeneities in units of Tesla
    b0noise = kspace_get_b0_noise(params, xygrid)

    # Get gradient values and expected k-space locations as a function of time
    gradients, kspace = kspace_make_pulse_sequence(params)

    # Prepare figure
    f = kspace_prepare_figure()

    # Initialize a list to store frames for creating a movie
    M = []

    # Precompute spins
    spins = kspace_precompute(params, gradients, xygrid, b0noise)

    # Main loop: simulate k-space data acquisition
    T = len(gradients['T'])
    progress_bar = tqdm(total=T, disable=params.get('showProgress', False))

    # Add a flag to control the loop
    stop_simulation = False

    # Print instructions for the user
    print("Press 'q' to quit the simulation.")

    # Define a keypress listener
    def on_press(key):
        nonlocal stop_simulation
        try:
            if key.char == 'q':  # Check if 'q' is pressed
                print("Simulation interrupted by user.")
                stop_simulation = True
        except AttributeError:
            pass

    # Start the keypress listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    for t in range(T):
        # Check if the simulation should stop
        if stop_simulation:
            break

        # Calculate the change in spin phase at each discrete location
        spins = kspace_compute_one_point(params, gradients, xygrid, b0noise, spins, t)

        # Update the spin phase to account for the most recent step
        spins = kspace_get_current_basis_functions(spins)

        # Get signal for this time point (fill 1 point in k-space)
        kspace = kspace_get_current_signal(kspace, t, im, spins, params)

        # Show plots
        kspace, M = kspace_show_plots(f, spins, gradients, kspace, im, params, t, b0noise, M)

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()

    # Stop the keypress listener
    listener.stop()
    
    res = (params, im, kspace, spins, gradients)

    return res
