#!/usr/bin/env python3
import os
import argparse
import subprocess
import time

def run_command(command):
    """Run a command and print its output in real-time"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Print output in real-time
    if process.stdout:
        for line in process.stdout:
            print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run self-modeling comparison experiments")
    parser.add_argument("--max_epochs", type=int, default=150,
                        help="Maximum number of epochs to train each model")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of training samples to generate")
    parser.add_argument("--base_dir", type=str, default="checkpoints/self_modeling_comparison",
                        help="Base directory for saving checkpoints and visualizations")
    parser.add_argument("--target_accuracy", type=float, default=None,
                        help="Target test accuracy to stop training (e.g., 0.8 for 80%%)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots after generating them")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Run training if not skipped
    if not args.skip_training:
        print("=== Running Self-Modeling Comparison Training ===")
        training_command = (
            f"python compare_self_modeling_locations.py "
            f"--max_epochs {args.max_epochs} "
            f"--samples {args.samples} "
            f"--base_dir {args.base_dir} "
            f"--seed {args.seed}"
        )
        
        # Add target accuracy if specified
        if args.target_accuracy is not None:
            training_command += f" --target_accuracy {args.target_accuracy}"
        result = run_command(training_command)
        if result != 0:
            print("Training failed. Check the error messages above.")
            return
    
    # Run visualization
    print("\n=== Generating Visualizations ===")
    viz_command = f"python visualize_self_modeling_results.py --base_dir {args.base_dir}"
    result = run_command(viz_command)
    if result != 0:
        print("Visualization failed. Check the error messages above.")
        return
    
    # Show plots if requested
    if args.show_plots:
        print("\n=== Displaying Plots ===")
        # Use matplotlib to display the plots
        import matplotlib.pyplot as plt
        import glob
        
        plot_files = glob.glob(os.path.join(args.base_dir, "*.png"))
        for plot_file in plot_files:
            plt.figure()
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(plot_file))
        
        plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Results saved to {args.base_dir}")

if __name__ == "__main__":
    main()