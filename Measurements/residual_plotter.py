import numpy as np
import matplotlib.pyplot as plt

def compute_norm(values):
    """Compute the Euclidean norm (L2 norm) of a list of numeric values."""
    return np.linalg.norm(values)

import numpy as np
import matplotlib.pyplot as plt

def compute_norm(values):
    """Compute the Euclidean norm (L2 norm) of a list of numeric values."""
    return np.linalg.norm(values)

def plot_line_norms(file_path, batch_size=2000):
    """Read the file, compute norms of numeric values in each line,
    compute the mean of every `batch_size` norms, and plot the results."""
    norms = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line into numeric values
            values = line.strip().split()
            
            try:
                # Convert the values to float
                numeric_values = np.array([float(value) for value in values])
                
                # Compute the norm of these numeric values
                norm = compute_norm(numeric_values)
                norms.append(norm)
            except ValueError:
                # If conversion fails, skip the line or handle it as needed
                print(f"Skipping line due to conversion error: {line.strip()}")
    
    # Compute means of every `batch_size` norms
    means = []
    for i in range(0, len(norms), batch_size):
        batch = norms[i:i + batch_size]
        means.append(np.mean(batch))
    
    # Plotting
    means = [loc_value if loc_value < 1 else 1 for loc_value in means]
    plt.figure(figsize=(20, 6))
    plt.plot(means, marker='o', linestyle='-', color='b')
    plt.title('Residual Norm (Each point is mean over 2000 timesteps)')
    plt.xlabel('Timesteps * 2000')
    plt.ylabel('Residual')
    #plt.ylim(-0.5, 1.5)  # Setting the y-axis limit to 1.5
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Replace 'residual_log_pincher2.txt' with the path to your text file
    plot_line_norms('residual_log_pincher2_2024-8-9_16-17-41.txt')