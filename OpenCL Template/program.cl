// #define GLINTEROP

uint GetBit ( __global uint* buffer, uint x, uint y)
{
	return (buffer[y * 512 + (x >> 5)] >> (int)(x & 31)) & 1U;
}

#ifdef GLINTEROP
__kernel void device_function( write_only image2d_t a, float t )
#else
__kernel void device_function( __global uint* buffer, float time, uint pw, uint ph, uint pwph, __global uint* pattern, __global uint* second )
#endif
{
	for (int i = 0; i < pwph; i++)
	{
		pattern[i] = 0;
	}
	
	// process all pixels, skipping one pixel boundary
    uint w = pw * 32, h = ph;
    for (uint y = 1; y < h - 1; y++) for (uint x = 1; x < w - 1; x++)
    {
        // count active neighbors
        uint n = GetBit(second, x - 1, y - 1) + GetBit(second, x, y - 1) + GetBit(second, x + 1, y - 1) + GetBit(second, x - 1, y) + GetBit(second, x + 1, y) + GetBit(second, x - 1, y + 1) + GetBit(second, x, y + 1) + GetBit(second, x + 1, y + 1);
        if ((GetBit(second, x, y) == 1 && n == 2) || n == 3) 
		{
			//BitSet(x, y);			
			pattern[y * 512 + (x >> 5)] |= 1U << (int)(x & 31);
		}
	}
	for (int i = 0; i < pwph; i++)
	{
		second[i] = pattern[i];
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
*/
}
