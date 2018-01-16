#define GLINTEROP

uint GetBit ( __global uint* buffer, uint x, uint y)
{
	return (buffer[y * 512 + (x >> 5)] >> (int)(x & 31)) & 1U;
}

__kernel void device_function( write_only image2d_t a, uint pw, uint ph, uint amountOfCells, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset )
{
	int id = get_global_id(0);

    if (id >= amountOfCells)
	{
		return;
	}

	// pw is het aantal uints dat het level breed is.
	uint x = id % (pw * 32);
	uint y = id / (pw * 32);

	// x2 is juiste uint	
	uint x2 = x / 32;

	pattern[x2] |= 1U << (int)(x & 31);


	if (x < xoffset || x > (xoffset + 511) || y < yoffset || y > (yoffset + 511))
	{
		return;
	}

	int2 pos = (int2)((int)x - xoffset, (int)y - yoffset);
	float4 col = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
	write_imagef(a, pos, col);
}

__kernel void copy_data(int pw, int amountOfCells, __global uint* pattern, __global uint* second)
{
	int id = get_global_id(0);

    if (id >= amountOfCells)
	{
		return;
	}

	uint x = id % (pw * 32);
	uint x2 = x / 32;
	second[x2] = pattern[x2];
}