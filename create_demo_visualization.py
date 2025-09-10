"""
Mock visualization script to demonstrate trajectory visualization UI layout.

This script creates a visual representation of what the trajectory visualization
feature would look like in the pygame window, showing the UI elements and
trajectory trails.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_trajectory_visualization_mockup():
    """Create a mockup of the trajectory visualization feature."""
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Set background color to simulate pygame window
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Draw map obstacles (rectangles)
    obstacles = [
        patches.Rectangle((1, 1), 2, 1, linewidth=1, edgecolor='black', 
                         facecolor='#141E14', alpha=0.5),
        patches.Rectangle((6, 4), 1.5, 2, linewidth=1, edgecolor='black', 
                         facecolor='#141E14', alpha=0.5),
        patches.Rectangle((3, 6), 3, 0.8, linewidth=1, edgecolor='black', 
                         facecolor='#141E14', alpha=0.5),
    ]
    
    for obstacle in obstacles:
        ax.add_patch(obstacle)
    
    # Robot trajectory (blue trail)
    robot_trajectory_x = [0.5, 1.2, 2.1, 3.2, 4.5, 5.8, 6.9, 7.8, 8.5]
    robot_trajectory_y = [2.0, 2.3, 2.8, 3.2, 3.8, 4.1, 3.9, 3.5, 3.0]
    ax.plot(robot_trajectory_x, robot_trajectory_y, color='#0064FF', linewidth=3, 
            label='Robot Trajectory', alpha=0.8)
    
    # Current robot position (circle)
    ax.scatter(robot_trajectory_x[-1], robot_trajectory_y[-1], color='#0000C8', 
              s=200, zorder=5, label='Robot')
    
    # Pedestrian trajectories (red trails)
    ped1_x = [1.0, 1.3, 1.8, 2.5, 3.1, 3.9, 4.6, 5.2, 5.7]
    ped1_y = [0.5, 1.0, 1.6, 2.1, 2.6, 3.0, 3.3, 3.5, 3.6]
    ax.plot(ped1_x, ped1_y, color='#FF6464', linewidth=2, alpha=0.7)
    ax.scatter(ped1_x[-1], ped1_y[-1], color='#FF3232', s=80, zorder=5)
    
    ped2_x = [8.5, 8.2, 7.8, 7.3, 6.7, 6.0, 5.3, 4.8, 4.4]
    ped2_y = [6.5, 6.2, 5.8, 5.4, 5.0, 4.6, 4.3, 4.1, 4.0]
    ax.plot(ped2_x, ped2_y, color='#FF6464', linewidth=2, alpha=0.7)
    ax.scatter(ped2_x[-1], ped2_y[-1], color='#FF3232', s=80, zorder=5)
    
    # Ego pedestrian trajectory (magenta trail)
    ego_x = [9.0, 8.7, 8.3, 7.8, 7.2, 6.5, 5.8, 5.2, 4.7]
    ego_y = [1.0, 1.5, 2.1, 2.8, 3.4, 3.9, 4.3, 4.6, 4.8]
    ax.plot(ego_x, ego_y, color='#C800C8', linewidth=3, alpha=0.8)
    ax.scatter(ego_x[-1], ego_y[-1], color='#6C4675', s=150, zorder=5, 
              marker='s', label='Ego Pedestrian')
    
    # Add grid
    ax.grid(True, alpha=0.3, color='#C8C8C8')
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 9))
    
    # Add text overlays to simulate pygame UI
    # Main status text (top-left)
    status_text = [
        "step: 142",
        "scaling: 15",
        "target fps: 58.2/60.0",
        "speedup: 5.8x",
        "x-offset: -2.34",
        "y-offset: -1.67",
        "(Press h for help)"
    ]
    
    # Create text box background
    status_box = FancyBboxPatch((0.2, 6.5), 2.8, 1.3,
                               boxstyle="round,pad=0.1",
                               facecolor='black', alpha=0.7,
                               edgecolor='none')
    ax.add_patch(status_box)
    
    # Add status text
    for i, text in enumerate(status_text):
        ax.text(0.3, 7.6 - i*0.15, text, color='white', fontsize=8, 
               fontfamily='monospace')
    
    # Playback status text (bottom-right)
    playback_text = [
        "Frame: 142/500",
        "Playing: Yes", 
        "Speed: 2.0x",
        "Trajectories: ON",
        "Trail Length: 100"
    ]
    
    # Create playback text box
    playback_box = FancyBboxPatch((7.0, 0.2), 2.8, 1.0,
                                 boxstyle="round,pad=0.1", 
                                 facecolor='black', alpha=0.7,
                                 edgecolor='none')
    ax.add_patch(playback_box)
    
    # Add playback text
    for i, text in enumerate(playback_text):
        ax.text(7.1, 1.0 - i*0.15, text, color='white', fontsize=8,
               fontfamily='monospace')
    
    # Help text (top-right)
    help_text = [
        "--- Playback Controls ---",
        "Space: Play/pause",
        "Period (.): Next frame", 
        "Comma (,): Previous frame",
        "n: First frame",
        "m: Last frame",
        "--- Trajectory Controls ---",
        "v: Toggle trajectories",
        "b: Increase trail length",
        "c: Decrease trail length",
        "x: Clear trajectories"
    ]
    
    # Create help text box
    help_box = FancyBboxPatch((6.8, 4.8), 3.0, 2.8,
                             boxstyle="round,pad=0.1",
                             facecolor='black', alpha=0.7, 
                             edgecolor='none')
    ax.add_patch(help_box)
    
    # Add help text
    for i, text in enumerate(help_text):
        ax.text(6.9, 7.4 - i*0.2, text, color='white', fontsize=7,
               fontfamily='monospace')
    
    # Title
    ax.set_title('RobotSF Interactive Playback - Trajectory Visualization Feature', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend for trajectories
    legend_elements = [
        plt.Line2D([0], [0], color='#0064FF', linewidth=3, label='Robot Trajectory'),
        plt.Line2D([0], [0], color='#FF6464', linewidth=2, label='Pedestrian Trajectories'), 
        plt.Line2D([0], [0], color='#C800C8', linewidth=3, label='Ego Pedestrian Trajectory')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add annotation arrow pointing to trajectory
    ax.annotate('Entity movement trails\nshow path history', 
                xy=(5.0, 3.8), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Create and save the trajectory visualization mockup."""
    print("Creating trajectory visualization mockup...")
    
    fig = create_trajectory_visualization_mockup()
    
    # Save the figure
    output_path = 'trajectory_visualization_demo.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Mockup saved as: {output_path}")
    print("\nThis image shows what the trajectory visualization feature looks like:")
    print("- Blue trails show robot movement history")
    print("- Red trails show pedestrian movement history") 
    print("- Magenta trails show ego pedestrian movement history")
    print("- UI overlays show trajectory controls and status")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()