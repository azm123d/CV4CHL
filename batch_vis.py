import sys
import subprocess
from pathlib import Path

def main():
    indices = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
    dataset_dir = Path("dataset")
    vis_script = "vis.py"

    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        sys.exit(1)

    for index in indices:
        folder_name = f"{index:04d}"
        folder_path = dataset_dir / folder_name
        
        if not folder_path.exists():
            print(f"Warning: Directory not found: {folder_path}")
            continue
            
        for subfolder in folder_path.iterdir():
            if subfolder.is_dir():
                print(f"Processing: {subfolder}")
                # 将输出结果直接保存在 vis_output 下的序号目录中
                output_dir = Path("vis_output") / folder_name
                
                cmd = [
                    "python", vis_script,
                    "--input_dir", str(subfolder),
                    "--output_dir", str(output_dir),
                    "--mode", "video"
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing vis.py on {subfolder}: {e}")

if __name__ == "__main__":
    main()
