import sys
import os
import glob
import stitcher

def stitchImages(path_to_files, basename, output_path):

	os.chdir(path_to_files)
	img_files = glob.glob(basename + "*")
	img_count = len(img_files)

	img_stitcher = stitcher.Stitcher()
	img_stitcher.init(img_files[0])

	print( "Stitching image files from " + path_to_files + " ... " )
	for i in range(0, img_count):
		print( "\t --- Stitching in image {img} --- ".format(img=img_files[i]) )
		
		img_path = os.path.join(path_to_files, img_files[i])

		img_stitcher.stitch(img_path)

	print( "Stitching image files from " + path_to_files + " ... DONE " )

	out_file = os.path.join( output_path, "{bn}_full.0001.tiff".format(bn=basename))
	print( "Outputing to " + out_file )

	img_stitcher.write(out_file)

def main():
	path_to_files = sys.argv[1]
	basename = sys.argv[2]
	output_path = sys.argv[3]

	output_path = os.path.abspath(output_path)
	path_to_files = os.path.abspath(path_to_files)

	scripts = stitchImages(path_to_files, basename, output_path)

main()
