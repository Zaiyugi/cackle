#include <iostream>
#include <OpenImageIO/imageio.h>

#include "Stitcher.h"

OIIO_NAMESPACE_USING

namespace akasha
{

void Stitcher::checkImage(const char *file_path, int &w, int &h, int &d)
{
    ImageInput *in = ImageInput::create(file_path);
    if(!in) return;

    ImageSpec spec;
    in->open(file_path, spec);
    w = spec.width;
    h = spec.height;
    d = spec.nchannels;
    in->close();
    delete in;
}

void Stitcher::readImage(const char *file_path, float *output)
{
    ImageInput *in = ImageInput::create(file_path);
    if(!in) return;

    ImageSpec spec;
    in->open(file_path, spec);
    int w = spec.width;
    int h = spec.height;
    int d = spec.nchannels;

    in->read_image(TypeDesc::FLOAT, output);

    in->close ();
    delete in;
}

void Stitcher::init(const char* file_path)
{
    checkImage(file_path, width, height, depth);
    printf("Image Dimensions: %d x %d x %d \n", width, height, depth);

    image_data = std::unique_ptr<float[]>(new float[width*height*depth]);
}

void Stitcher::stitch(const char* file_path)
{
    std::unique_ptr<float[]> input(new float[width * height * depth]);

    readImage(file_path, input.get());

    for(int i = 0; i < width*height; i++)
    {
        // If the current pixel was rendered, copy into image_data
        if(input[i * depth + 3] != 0.0)
        {
            image_data[i * depth    ] = input[i * depth];
            image_data[i * depth + 1] = input[i * depth + 1];
            image_data[i * depth + 2] = input[i * depth + 2];
            image_data[i * depth + 3] = input[i * depth + 3];
        }
    }
}

void Stitcher::write(const char* output_path)
{
    ImageOutput* out = ImageOutput::create(output_path);
    if(!out)
    {
        std::cout << "Unable to open file for writing: " << output_path << std::endl;
    } else
    {
        ImageSpec spec( width, height, depth, TypeDesc::FLOAT);
        out->open(output_path, spec);
        out->write_image(TypeDesc::FLOAT, image_data.get());
        out->close();
        delete out;
    }

}

} // namespace menrva
