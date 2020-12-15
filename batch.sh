#!/bin/bash
###
# usage: ./batch.sh "/path/to/folder/with/input/video" "/path/to/output/video"
# calls jumpcutter on each file in the input dir
# doesn't differentiate non-video files etc so it will give garbage to jumpcutter.py if it's present
# plan to include this functionality in the main py script at some point
###
for i in "$1"/*; do
    name=$(basename "$i")
    python3 jumpcutter.py --input_file "$i" --output_file "$2/$name"

done