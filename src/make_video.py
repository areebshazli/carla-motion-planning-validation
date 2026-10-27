#!/usr/bin/env python3
"""Turn recorded frames into an MP4 without rerunning the simulation.

    pip install imageio imageio-ffmpeg
    python make_video.py
    python make_video.py --frames controller_output\\frames --fps 20 --out run.mp4
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "controller_output", "frames"),
                    help="folder with the numbered PNG frames")
    ap.add_argument("--fps", type=int, default=20, help="frames per second (20 or 10 as recorded)")
    ap.add_argument("--out", default=os.path.join(HERE, "controller_output", "run.mp4"))
    args = ap.parse_args()

    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("Please run:  pip install imageio imageio-ffmpeg   and then try again.")

    frames = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if not frames:
        sys.exit("No PNG frames found in %s" % args.frames)
    print("Encoding %d frames at %d fps ..." % (len(frames), args.fps))
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for i, f in enumerate(frames):
        writer.append_data(imageio.imread(f))
        if i % 200 == 0:
            print("  %d / %d" % (i, len(frames)))
    writer.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
