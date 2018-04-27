#include "Image.h"
#include <iostream>
using namespace std;
using namespace lux;

float Image::interpolatedValue( float x, float y, int c ) const
{
   int ix, iy, iix, iiy;
   float wx, wy, wwx, wwy;
   interpolationCoefficients( x, y, wx, wwx, wy, wwy, ix, iix, iy, iiy );
   float v = value(  ix,  iy, c ) *  wx *  wy
           + value( iix,  iy, c ) * wwx *  wy
	   + value(  ix, iiy, c ) *  wx * wwy
	   + value( iix, iiy, c ) * wwx * wwy;
   return v;
}


std::vector<float> Image::interpolatedPixel( float x, float y ) const
{
   int ix, iy, iix, iiy;
   float wx, wy, wwx, wwy;
   interpolationCoefficients( x, y, wx, wwx, wy, wwy, ix, iix, iy, iiy );
   std::vector<float> pix;
   for( int c=0;c<Depth();c++ )
   {
      float v = value(  ix,  iy, c ) *  wx *  wy
              + value( iix,  iy, c ) * wwx *  wy
	      + value(  ix, iiy, c ) *  wx * wwy
	      + value( iix, iiy, c ) * wwx * wwy;
      pix.push_back( v );
   }
   return pix;
}

void imageLinearInterpolation( float x, float dx, int nx, int& i, int& ii, float& w, float& ww, bool isperiodic )
{
    float r = x/dx;
    i  =  r;
    if( !isperiodic )
    {
    if( i >= 0 && i < nx )
    {
       ii = i + 1;
       ww = r-i;
       w = 1.0 - w;
       if(  ii >= nx )
       {
	  ii = nx-1;
       }
    }
    else
    {
       i = ii = 0;
       w = ww = 0;
    }
    }
    else
    {
       ww = r-i;
       while( i < 0 ){ i += nx; }
       while( i >= nx ){ i -= nx; }

       ii = i+1;
       w = 1.0 - w;
       if( ii >= nx ){ ii -= nx; }
    }
}




void Image::interpolationCoefficients( float x, float y, 
                                      float& wx, float& wwx,
				      float& wy, float& wwy,
				      int& ix, int& iix,
				      int& iy, int& iiy 
			  	    ) const
{
   imageLinearInterpolation( x, 1.0/Width(), Width(), ix, iix, wx, wwx, false );
   imageLinearInterpolation( y, 1.0/Height(), Height(), iy, iiy, wy, wwy, false );
}


void lux::setPixel( Image& img, int x, int y, std::vector<float>& value )
{
   size_t nb = ( value.size() < img.Depth() ) ? value.size() : img.Depth();
   for( size_t i=0;i<nb;i++ )
   {
      img.value( x, y, i ) = value[i];
   }
}



void lux::setPixel( Image& img, int x, int y, float r, float g, float b, float a )
{
   if(img.Depth()>0){ img.value(x,y,0) = r; }
   if(img.Depth()>1){ img.value(x,y,1) = g; }
   if(img.Depth()>2){ img.value(x,y,2) = b; }
   if(img.Depth()>3){ img.value(x,y,3) = a; }
}
