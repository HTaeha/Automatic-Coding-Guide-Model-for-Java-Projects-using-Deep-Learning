#!/bin/bash
input_file_name=guava

cd class_file
cd ${input_file_name}_class_collected
for f in *; do
	file_name=$(basename $f)
	file_no_extension="${file_name%.*}"
	javap -c $file_name &> ../../disassemble_file/${input_file_name}_disassemble_file/$file_no_extension.txt
done
