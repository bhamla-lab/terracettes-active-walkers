import numpy as np
import matplotlib.pyplot as plt
import noise
from scipy.signal import find_peaks, peak_prominences
from statsmodels.tsa.stattools import acf

def generate_H(scale, resolution, sigma, noise_scale, plot=False):

    # define the surface size (in units of resolution)
    size = scale * resolution

    # Create a meshgrid to represent the surface
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y)

    # Calculate the elevation based on the y-coordinate and the slope angle
    slope_angle_radians = np.radians(sigma)
    heightmap_slope = np.tan(slope_angle_radians) * y
    offset_lon = np.tan(slope_angle_radians) * (size-1)
    offset_lat = 0

    # Store offsets for periodic boundary condition
    offsets = np.array([offset_lat, offset_lon])

    # Generate Perlin noise
    octaves = 10  # Number of levels of detail
    persistence = 0.5  # sigma of each octave
    lacunarity = 2.0  # Frequency of each octave
    perlin_noise = np.zeros_like(heightmap_slope)
    for i in range(size):
        for j in range(size):
            perlin_noise[i][j] = noise.pnoise2(i / size,  
                                            j / size, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=1, 
                                            repeaty=1, 
                                            base=311)

    # Normalize Perlin noise to match the elevation range
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())

    # Add Perlin noise to the elevation map
    H = heightmap_slope + (perlin_noise * noise_scale)

    if plot: 
        # Create the 3D surface plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, H, cmap='rainbow', alpha=0.8)
        ax.contour3D(x, y, H, 25, cmap='binary')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Surface Plot of H')
        plt.show()
        
    return x, y, H, size, offsets 

# build a spherical brush 
def generate_F(radius):
    size = 2 * radius
    kernel = np.zeros((size, size))

    center = radius - 0.5  # Centering the brush
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            if distance < radius:
                normalized_distance = distance / radius
                alpha = np.cos((normalized_distance * np.pi) / 2)  # Using cosine for spherical transition
                kernel[i, j] = alpha

    return kernel

# build a circular brush 
def generate_D(radius):
    size = 2 * radius
    kernel = np.zeros((size, size))

    center = radius - 0.5  # Centering the brush
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            if distance < radius:
                alpha = 1 
                kernel[i, j] = alpha

    return kernel

def accumulate_traces(H, kernel, line_points, alpha, eta):

    sizex, sizey = H.shape
    brush_sizex, brush_sizey = kernel.shape
    brush_radius = brush_sizex // 2

    accumulated_effect = np.zeros((sizex, sizey))

    for k in range(len(line_points)-1):
        posx, posy = line_points[k+1,:] + eta*(np.random.rand(2)-0.5)
        for i in range(brush_sizex):
            for j in range(brush_sizey):
                canvas_y = int((posx - brush_radius + i) % sizey)
                canvas_x = int((posy - brush_radius + j) % sizex)
                
                accumulated_effect[canvas_x, canvas_y] += kernel[i, j] * alpha
    
    H += accumulated_effect
    
    return H, accumulated_effect

def compute_costs(agent_pos, B, delta, L, trampled, R, H_size, offsets, resolution, cumulative=False):
    """
    Computes the radial costs and generates a meshgrid for visualization.
    """
    # Generate 360-degree angles to cover all directions
    angles = np.linspace(0, 2*np.pi, B, endpoint=False)

    # Precompute radial steps for the maximum radius
    radial_steps = np.linspace(0, L, int(L/delta)) # Radial points along each direction

    # Create the meshgrid of angles and radial distances
    angles_mesh, radial_mesh = np.meshgrid(angles, radial_steps, indexing='ij')

    # Convert the polar meshgrid to Cartesian coordinates
    x_mesh = agent_pos[0] + radial_mesh * np.cos(angles_mesh)
    y_mesh = agent_pos[1] + radial_mesh * np.sin(angles_mesh)

    # Reduce to int for indexing
    x_indices = np.round(x_mesh).astype(int)
    y_indices = np.round(y_mesh).astype(int)

    # Height corrections for periodic boundary condition
    offset_lat, offset_lon = offsets
    applied_offsets = np.zeros_like(x_indices, dtype=float)
    
    # Apply offset corrections to the heights
    applied_offsets += np.where(y_indices % H_size < y_indices, offset_lon, 0)
    applied_offsets -= np.where(y_indices % H_size > y_indices, offset_lon, 0)
    applied_offsets += np.where(x_indices % H_size < x_indices, offset_lat, 0)
    applied_offsets -= np.where(x_indices % H_size > x_indices, offset_lat, 0)
    
    # Wrap x and y indices for periodic boundary conditions
    x_indices %= H_size
    y_indices %= H_size
    
    # Compute gradient cost with corrected heights
    heights = (trampled[y_indices, x_indices] + applied_offsets)/resolution
    # A small region of R, nearest the agent, is excluded from consideration, since the agents previous depletion would otherwise impact their forward movement cost. I.e., we ignore the R over the first footstep. 
    resource_exclusion = radial_mesh > int(resolution * 2.1)
    
    #E(phi)
    if cumulative: 
        gradient_diff = np.abs(np.diff(heights, axis=1))  # Compute differences
        gradient = np.cumsum(np.hstack([np.zeros((gradient_diff.shape[0], 1)), gradient_diff]), axis=1)  # Prepend zero and accumulate
        depletion = np.cumsum((1 - R[y_indices, x_indices]) * resource_exclusion, axis=1)  # Resource depletion cost       
    #e_k(phi)
    else:  
        gradient = np.sum(np.abs(np.diff(heights, axis=1)), axis=1)  # Gradient cost along radial lines
        depletion = np.sum((1 - R[y_indices, x_indices]) * resource_exclusion, axis=1)  # Resource depletion cost

    return angles_mesh, radial_mesh, gradient, depletion

def pick_direction(agent_pos, beta, angles_mesh, radial_mesh, gradient, depletion, w_h, w_v, delta, H_size):

    # Sum energy components to calculate total energy cost E 
    E =  (w_h*delta*depletion) + (w_v*gradient)
    
    # Normalize the total energy cost
    E_tilde = E/np.max(E)
    E_tilde = np.nan_to_num(E_tilde, nan=0.0) # if np.max(total_costs) happens to be zero
    
    # Scale energy costs using the exponential 
    S = np.exp(-beta * E_tilde)

    # generate discrete probability distribution probability of direction choice is based on exponential cost
    P = S / np.sum(S)
    ind = np.random.choice(len(P), p=P)
    angle = angles_mesh[ind,0]
    xline = agent_pos[0] + radial_mesh[ind,:] * np.cos(angle)
    yline = agent_pos[1] + radial_mesh[ind,:] * np.sin(angle)
    agent_pos = (xline[-1]%H_size, yline[-1]%H_size)
    
    # calculate the shannon entropy of P 
    entropy = -np.sum(P * np.log(P + np.finfo(float).eps))  # Add epsilon to avoid log(0)
    
    return agent_pos, np.array([xline%H_size, yline%H_size]).T, angle, entropy 

def fingerprint(erosion, omega, ground_truth=None, plot=False):
    # Using technique outlined in Bazen & Gerez, 2002  

    # Compute the gradient in the x and y directions
    G_y, G_x = np.gradient(erosion)

    # initialize arrays 
    G_x_B = np.zeros([len(range(0, erosion.shape[0], omega)), len(range(0, erosion.shape[1], omega))])
    G_y_B = np.zeros([len(range(0, erosion.shape[0], omega)), len(range(0, erosion.shape[1], omega))])
    #coherence = np.zeros([len(range(0, total_erosion.shape[0], omega)), len(range(0, total_erosion.shape[1], omega))])
    #Vi = np.zeros([len(range(0, total_erosion.shape[0], omega)), len(range(0, total_erosion.shape[1], omega))])
    #Vj = np.zeros([len(range(0, total_erosion.shape[0], omega)), len(range(0, total_erosion.shape[1], omega))])
    G_xx = np.zeros([len(range(0, erosion.shape[0], omega)), len(range(0, erosion.shape[1], omega))]) 
    G_yy = np.zeros([len(range(0, erosion.shape[0], omega)), len(range(0, erosion.shape[1], omega))])
    G_xy = np.zeros([len(range(0, erosion.shape[0], omega)), len(range(0, erosion.shape[1], omega))])

    # Loop through the array in ω×ω blocks
    for i, omegai in enumerate(range(0, erosion.shape[0], omega)):
        for j, omegaj in enumerate(range(0, erosion.shape[1], omega)):
            # Acquire gradients within sqaure windows of size omega
            G_x_box = G_x[omegai:omegai + omega, omegaj:omegaj + omega]
            G_y_box = G_y[omegai:omegai + omega, omegaj:omegaj + omega]
            # Compute the average gradient (for visualization)
            G_x_B[i,j] = np.mean(G_x_box)
            G_y_B[i,j] = np.mean(G_y_box)
            # Estimate the variances and crossvariances of G_x and G_y over each window
            G_xx[i,j] = np.mean(G_x_box**2) # no more negatives? Do we lose direction? 
            G_yy[i,j] = np.mean(G_y_box**2)
            G_xy[i,j] = np.mean(G_x_box*G_y_box)
            
    Vi = (0.5*(G_xx - G_yy)) - (0.5*np.sqrt(((G_xx - G_yy)**2) + (4*(G_xy**2))))
    Vj = G_xy

    # Coherence calculation
    #coherence = np.sqrt(((G_xx - G_yy)**2) + (4*(G_xy**2))) / (G_xx + G_yy)
    coherence = np.where(G_xx + G_yy == 0, 0, np.sqrt(((G_xx - G_yy)**2) + (4*(G_xy**2))) / (G_xx + G_yy))


    theta = np.arctan(Vj/Vi)%(np.pi)
    # Create grid for plotting at lower resolution (center of each block)
    Y, X = np.mgrid[0:erosion.shape[0], 0:erosion.shape[1]]
    Y_downsampled = Y[omega//2::omega, omega//2::omega]
    X_downsampled = X[omega//2::omega, omega//2::omega]
    mean_gradient = np.mean(np.sqrt(G_y**2 + G_x**2))
    mean_coherence = np.mean(coherence)
    
    if plot: 

        # Plot gradient field (for clarity)
        plt.figure(figsize=(10, 10))
        plt.imshow(erosion, cmap='gray', interpolation='nearest', origin='lower')
        plt.colorbar(label='Total Erosion')
        plt.title(f'Gradient Field (B = {omega}x{omega}). Mean Gradient: {mean_gradient}')
        plt.quiver(X_downsampled, Y_downsampled, -G_x_B, -G_y_B, color='k', scale=50)
        plt.show()

        # Plot orientation field 
        plt.figure(figsize=(10, 10))
        plt.imshow(erosion, cmap='gray', interpolation='nearest', origin='lower')
        plt.colorbar(label='Total Erosion')
        plt.title(f'Orientation Field (B = {omega}x{omega}). Mean Coherence: {mean_coherence}')
        plt.quiver(X_downsampled, Y_downsampled, coherence*np.cos(theta), coherence*np.sin(theta), color='k', scale=40, headaxislength=0, headlength=0, pivot='mid')
        plt.show()

        # Plot coherence field 
        # plt.figure(figsize=(10, 10))
        # plt.imshow(coherence, origin='lower', vmin=0, vmax=1, cmap='grey')
        # plt.title(f'Mean Coherence: {mean_coherence}')
        # plt.colorbar(label='Anisotropic Coherence')
        # plt.show()

    if isinstance(ground_truth, np.ndarray):
        # Assuming orientation_field and ground_truth are in radians and within the range [0, π]
        difference = np.abs(theta - ground_truth)
        # Ensure the difference is within the range [0, π]
        difference = np.where(difference > np.pi/2, np.pi - difference, difference)
        mean_difference = np.mean(difference)
        
        if plot: 
            # Plot the difference for visual comparison
            plt.figure(figsize=(10, 10))
            plt.imshow(difference, cmap='hot', origin='lower')
            plt.colorbar(label='Orientation Difference')
            plt.title(f'Orientation Field Difference. Mean={mean_difference}')
            plt.show()
        
        return theta, mean_gradient, mean_coherence, mean_difference 

    return theta, mean_gradient, mean_coherence

def compute_gradient_lines(elevation_map, num_paths, step_size=5, plot=False):
    M, N = elevation_map.shape

    # Compute the mean elevation
    mean_elevation = np.mean(elevation_map)

    # Plot the contour for the mean elevation and extract the path
    contours = plt.contour(np.arange(N), np.arange(M), elevation_map, levels=[mean_elevation])
    plt.close()
    # Extract the contour line corresponding to the mean elevation
    mean_contour_path = contours.collections[0].get_paths()[0]
    contour_coords = mean_contour_path.vertices  # Get x, y coordinates of the contour line
    contour_x = contour_coords[:, 0]
    contour_y = contour_coords[:, 1]

    paths = []  # List to store the paths for each starting point

    # Compute terrain gradient
    gradient_y, gradient_x = np.gradient(elevation_map)

    # Choose points along the contour to seed the gradient lines
    num_contour_points = len(contour_x)
    contour_spacing = np.linspace(0, num_contour_points - 1, num_paths + 2, dtype=int)
    contour_spacing = contour_spacing[1:-1]  # Remove the very first and last points

    for i in contour_spacing:
        # Initialize paths with the starting point on the contour
        path_up_x, path_up_y = [contour_x[i]], [contour_y[i]]  # For the upward path
        path_down_x, path_down_y = [contour_x[i]], [contour_y[i]]  # For the downward path
        current_x_up, current_y_up = contour_x[i], contour_y[i]
        current_x_down, current_y_down = contour_x[i], contour_y[i]

        num_steps = 1000  # Maximum number of steps per path

        # Start from the contour point and move **up** the gradient
        for _ in range(num_steps):
            grad_x = gradient_x[int(current_y_up), int(current_x_up)]
            grad_y = gradient_y[int(current_y_up), int(current_x_up)]

            # Move in the direction of the gradient (uphill)
            new_x_up = current_x_up + step_size * grad_x
            new_y_up = current_y_up + step_size * grad_y

            # Stop if the new point goes out of bounds (edges of the map)
            if new_x_up < 0 or new_x_up >= N-1 or new_y_up < 0 or new_y_up >= M-1:
                break

            # Append the new point to the upward path
            path_up_x.append(new_x_up)
            path_up_y.append(new_y_up)

            # Update current position
            current_x_up, current_y_up = new_x_up, new_y_up

        # Start from the contour point and move **down** the gradient
        for _ in range(num_steps):
            grad_x = gradient_x[int(current_y_down), int(current_x_down)]
            grad_y = gradient_y[int(current_y_down), int(current_x_down)]

            # Move in the opposite direction of the gradient (downhill)
            new_x_down = current_x_down - step_size * grad_x
            new_y_down = current_y_down - step_size * grad_y

            # Stop if the new point goes out of bounds (edges of the map)
            if new_x_down < 0 or new_x_down >= N or new_y_down < 0 or new_y_down >= M:
                break

            # Append the new point to the downward path
            path_down_x.append(new_x_down)
            path_down_y.append(new_y_down)

            # Update current position
            current_x_down, current_y_down = new_x_down, new_y_down

        # Combine both up and down paths (center contour -> down, then center -> up)
        full_path_x = np.array(path_down_x[::-1] + path_up_x[1:])  # Avoid duplicating the center point
        full_path_y = np.array(path_down_y[::-1] + path_up_y[1:])
        
        paths.append((full_path_x, full_path_y))

    if plot:
        # Plot the topographic contours and all gradient paths
        plt.figure(figsize=(10, 10))
        
        # Plot the central contour line (mean elevation contour)
        plt.plot(contour_x, contour_y, 'r', label='Mean Elevation Contour', linewidth=2)

        viridis = plt.cm.get_cmap('viridis', num_paths)
        for i, path in enumerate(paths):
            path_x, path_y = path
            plt.plot(path_x, path_y, c=viridis(i / num_paths), label=f'Path {i + 1}')  # Sample the color map

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gradient Paths Extending from Contour Line to Map Edges')
        plt.legend()
        plt.axis('equal')
        plt.show()

    return paths

# Example call:
# paths = compute_gradient_lines(elevation_map, num_paths=5, plot=True)


def compute_ac(total_erosion, paths, lag=101, resolution=10, plot=False):

    acf_out = np.zeros([len(paths), lag])
    if plot: 
        plt.figure(figsize=(6, 2))
    
        viridis = plt.cm.get_cmap('viridis', len(paths))

    # Compute autocorrelations for all paths and store them
    for i, path in enumerate(paths):
        temp_x, temp_y = path
        temp_x = np.round(np.array(temp_x)).astype(int)
        temp_y = np.round(np.array(temp_y)).astype(int)

        # Compute autocorrelation for the current path
        acf_result = acf(total_erosion[temp_y, temp_x], nlags=lag-1)
        acf_out[i, :] = acf_result

        if plot: 
            # Plot the individual autocorrelation signal with low alpha
            plt.plot(acf_out[i, :], c=viridis(i /  len(paths)), lw=2, alpha=0.5)

    # Compute the mean autocorrelation across all paths
    mean_acf = np.mean(acf_out, axis=0)

    # Find peaks in the mean autocorrelation
    peaks, _ = find_peaks(mean_acf)

    # Identify the first nonzero peak
    first_nonzero_peak = next((p for p in peaks if mean_acf[p] > 0), None)

    if first_nonzero_peak is not None:
        # Calculate the prominence of the first nonzero peak
        prominences = peak_prominences(mean_acf, peaks)[0]
        first_peak_prominence = prominences[peaks == first_nonzero_peak][0]
        correlation = mean_acf[first_nonzero_peak]
        
        if plot: 
            # Plot the mean autocorrelation signal
            plt.plot(mean_acf, c='k', lw=3, label='Mean')

            # Plot the first nonzero peak on the mean autocorrelation signal
            plt.scatter(first_nonzero_peak, mean_acf[first_nonzero_peak], color='red', zorder=5)
            
            # Update values based on resolution
            plt.xticks([0, 20, 40, 60, 80, 100], [0, 2, 4, 6, 8, 10], fontsize=16)
            plt.yticks(fontsize=16)
    
    else: 
        return np.nan, np.nan, np.nan

    if plot: 
        # Plot formatting
        #plt.title('Mean Autocorrelation with First Nonzero Peak')
        plt.xlabel(r'$k$', fontsize=18)
        plt.ylabel(r'$R$', rotation=0, labelpad=10, fontsize=18)
        #plt.legend(fontsize=16)
        plt.show()
        
    return correlation, first_peak_prominence, first_nonzero_peak

def gradient_analysis(deviation, omega, plot=False):
    # Using technique outlined in Bazen & Gerez, 2002  

    # Compute the gradient in the x and y directions
    G_y, G_x = np.gradient(deviation)

    # initialize arrays 
    G_x_omega = np.zeros([len(range(0, deviation.shape[0], omega)), len(range(0, deviation.shape[1], omega))])
    G_y_omega = np.zeros([len(range(0, deviation.shape[0], omega)), len(range(0, deviation.shape[1], omega))])
    G_xx = np.zeros([len(range(0, deviation.shape[0], omega)), len(range(0, deviation.shape[1], omega))]) 
    G_yy = np.zeros([len(range(0, deviation.shape[0], omega)), len(range(0, deviation.shape[1], omega))])
    G_xy = np.zeros([len(range(0, deviation.shape[0], omega)), len(range(0, deviation.shape[1], omega))])

    # Loop through the array in ω×ω blocks
    for i, omegai in enumerate(range(0, deviation.shape[0], omega)):
        for j, omegaj in enumerate(range(0, deviation.shape[1], omega)):
            # Acquire gradients within sqaure windows of size omega
            G_x_omegaox = G_x[omegai:omegai + omega, omegaj:omegaj + omega]
            G_y_omegaox = G_y[omegai:omegai + omega, omegaj:omegaj + omega]
            # Compute the average gradient (for visualization)
            G_x_omega[i,j] = np.mean(G_x_omegaox)
            G_y_omega[i,j] = np.mean(G_y_omegaox)
            # Estimate the variances and crossvariances of G_x and G_y over each window
            G_xx[i,j] = np.mean(G_x_omegaox**2)
            G_yy[i,j] = np.mean(G_y_omegaox**2)
            G_xy[i,j] = np.mean(G_x_omegaox*G_y_omegaox)
    
    Vi = (0.5*(G_xx - G_yy)) - (0.5*np.sqrt(((G_xx - G_yy)**2) + (4*(G_xy**2))))
    Vj = G_xy

    # Coherence calculation
    coherence = np.where(G_xx + G_yy == 0, 0, np.sqrt(((G_xx - G_yy)**2) + (4*(G_xy**2))) / (G_xx + G_yy))

    theta = np.arctan(Vj/Vi)%(np.pi)
    # Create grid for plotting at lower resolution (center of each block)
    Y, X = np.mgrid[0:deviation.shape[0], 0:deviation.shape[1]]
    Y_downsampled = Y[omega//2::omega, omega//2::omega]
    X_downsampled = X[omega//2::omega, omega//2::omega]
    mean_gradient = np.mean(np.sqrt(G_y**2 + G_x**2))
    mean_coherence = np.mean(coherence)
    
    if plot:
        # Plot gradient field (for clarity)
        plt.figure(figsize=(10, 10))
        plt.imshow(deviation, cmap='gray', interpolation='nearest', origin='lower')
        plt.colorbar(label='Total deviation')
        plt.title(f'Gradient Field (B = {omega}x{omega}). Mean Gradient: {mean_gradient}')
        plt.quiver(X_downsampled, Y_downsampled, -G_x_omega, -G_y_omega, color='k', scale=50)
        plt.show()

        # Plot orientation field 
        plt.figure(figsize=(10, 10))
        plt.imshow(deviation, cmap='gray', interpolation='nearest', origin='lower')
        plt.colorbar(label='Total deviation')
        plt.title(f'Orientation Field (B = {omega}x{omega}). Mean Coherence: {mean_coherence}')
        plt.quiver(X_downsampled, Y_downsampled, coherence*np.cos(theta), coherence*np.sin(theta), color='k', scale=40, headaxislength=0, headlength=0, pivot='mid')
        plt.show()

        #Plot coherence field 
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence, origin='lower', vmin=0, vmax=1, cmap='grey')
        plt.title(f'Mean Coherence: {mean_coherence}')
        plt.colorbar(label='Anisotropic Coherence')
        plt.show()

    return theta, mean_gradient, mean_coherence

# MSD function
def compute_msd(positions):
    N = len(positions)
    msd = np.zeros(N)
    for tau in range(1, N):
        diffs = positions[tau:] - positions[:-tau]
        squared_displacements = np.sum(diffs**2, axis=1)
        msd[tau] = np.mean(squared_displacements)
    return msd
