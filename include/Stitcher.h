#ifndef __STITCHER_H__
#define __STITCHER_H__

#include <memory>

namespace akasha
{

class Stitcher
{
public:
    
    Stitcher() : width(0), height(0), depth(0) {}

    ~Stitcher() {}

    void init(const char* file_path);
    void stitch(const char* file_path);
    void write(const char* output_path);

private:
    void checkImage(const char *file_path, int &w, int &h, int &d);
    void readImage(const char *file_path, float *output);
    
    int width, height, depth;

    std::unique_ptr<float[]> image_data;

};

} // namespace akasha

#endif
