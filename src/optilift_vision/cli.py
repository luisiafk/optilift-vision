import argparse
from rich.console import Console
from rich.panel import Panel

from .tracker.pose_extractor import LiftTracker
from .math_engine.kinematics import procces_barbell_kinematics
from .math_engine.metrics import find_sticky_point
from .render.video_overlay import VideoRenderer

console = Console()

def main():
    parser = argparse.ArgumentParser(description="OptiLift-Vision: Biomecanic math engine")
    parser.add_argument("--video", required=True, help="Path to the video that you want to analyze (.mp4)")
    parser.add_argument("--output", default="output_analyzed.mp4", help="Path to output video")

    args = parser.parse_args()

    console.print(Panel.fit("[bold cyan]OptiLift-Vision[/bold cyan] - Numerical biomechanics engine", border_style="cyan"))

    with console.status("[bold green]Extracting the joints and barbell trajectory"):
        tracker = LiftTracker(args.video)
        poses = tracker.procces_video()

    if not poses:
        console.print("[bold red][ERROR][/bold red]Cannot extract the poses on the vide. Make sure that the lifter is on his side and completely visible")
        return

    console.print(f"[green][/green] Complete the extraction: {len(poses)} proccesed frames")


    with console.status("[bold green] Applying cubic splines and calculate the deritative (v, a) ..."):
        timestamps = [p.timestamp for p in poses]
        barbell_positions = [p.barbell for p in poses]
        states = procces_barbell_kinematics(timestamps, barbell_positions)

    console.print("[green][green]Completed cinematic")

    with console.status("[bold green] Searching for the sticking point"):
        sp_result = find_sticky_point(states)

    if sp_result["detected"]:
        console.print(f"[bold yellow] Sticking Point founded on the second {sp_result['timestamp']:.2f} (Frame {sp_result['frame_idx']})[/bold yellow]")
    else:
        console.print("[yellow]! Sticking point not detected [/yellow]")

    with console.status(f"[bold green] Rendering video on {args.output}"):
        renderer = VideoRenderer(args.video, args.output)
        sp_frame = sp_result["frame_idx"] if sp_result["detected"] else 0
        renderer.render(poses, states, sp_frame)

    console.print(f"\n[bold green] Analysis completed succesfully[/bold green] Video saved in: [underline]{args.output}[/underline]")

if __name__ == "__main__":
    main()
