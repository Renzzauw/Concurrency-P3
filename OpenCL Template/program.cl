// #define GLINTEROP

// helper function for getting one bit from the secondary pattern buffer
uint GetBit( __global int* a, uint x, uint y) 
{ 
	return (a[y * 512 + (x >> 5)] >> (int)(x & 31)) & 1U; 
}

#ifdef GLINTEROP
__kernel void device_function( write_only image2d_t a, float t )
#else
__kernel void device_function( __global int* a, float t )
#endif
{

    // process all pixels, skipping one pixel boundary
    for (uint y = 1; y < 511; y++) for (uint x = 1; x < 511; x++)
    {
        // count active neighbors
        uint n = GetBit(a, x - 1, y - 1) + GetBit(a, x, y - 1) + GetBit(a, x + 1, y - 1) + GetBit(a, x - 1, y) +
                    GetBit(a, x + 1, y) + GetBit(a, x - 1, y + 1) + GetBit(a, x, y + 1) + GetBit(a, x + 1, y + 1);
        if ((GetBit(a, x, y) == 1 && n == 2) || n == 3) 
		{
			a[y * 512 + (x >> 5)] |= 1U << (int)(x & 31);
		}
    }
		a[256 * 512 + (256 >> 5)] = (255 << 16) + (255 << 8) + 255;
}
/*
#ifdef GLINTEROP
	int2 pos = (int2)(idx,idy);
	write_imagef( a, pos, (float4)(col * (1.0f / 16.0f), 1.0f ) );
#else
	int r = (int)clamp( 16.0f * col.x, 0.f, 255.f );
	int g = (int)clamp( 16.0f * col.y, 0.f, 255.f );
	int b = (int)clamp( 16.0f * col.z, 0.f, 255.f );
	a[id] = (r << 16) + (g << 8) + b;
#endif
}
*/