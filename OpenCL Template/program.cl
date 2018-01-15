#define GLINTEROP

uint GetBit ( __global uint* buffer, uint x, uint y)
{
	return (buffer[y * 512 + (x >> 5)] >> (int)(x & 31)) & 1U;
}

#ifdef GLINTEROP
__kernel void device_function( write_only image2d_t a, float time, uint pw, uint ph, uint pwph, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset )
#else
__kernel void device_function( __global uint* buffer, float time, uint pw, uint ph, uint pwph, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset )
#endif
{
	int id = get_global_id(0);

    if (id >= pwph)
	{
		return;
	}

	uint x = id % pw % 32;
	uint y = id / pw;
	pattern[id / 32] |= 1U << (int)(x & 31);

	second[id / 32] = pattern[id / 32];

#ifdef GLINTEROP
/*
	int idx = id % pw + id / 32;
	int idy = id / pw;
	if (idx < xoffset || idx > (xoffset + 512) || idy < yoffset || idy > (yoffset + 512))
	{
		return;
	}

	printf("idx: %i, idy: %i\n", idx, idy);

	int2 pos = (int2)(idx, idy);
	float4 col = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
	write_imagef(a, pos, col);
*/
/*
	int idx = id % 32;
	int idy = id / 32;
	int2 pos = (int2)(idx,idy);
	float4 col = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
	write_imagef( a, pos, col );
*/
#else
	int r = (int)clamp( 16.0f * col.x, 0.f, 255.f );
	int g = (int)clamp( 16.0f * col.y, 0.f, 255.f );
	int b = (int)clamp( 16.0f * col.z, 0.f, 255.f );
	buffer[id] = (r << 16) + (g << 8) + b;
#endif
}
