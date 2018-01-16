#define GLINTEROP

int GetBit ( __global uint* second, uint x, uint y, uint pw)
{
	uint id = y * (pw * 32) + x;
	uint x2 = x / 32 + y * pw;

	int i = second[x2] >> (int)(x & 31) & 1U;
	return i;

	//return (second[y * 512 + (x >> 5)] >> (int)(x & 31)) & 1U;
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
	uint x2 = x / 32 + y * pw;

	pattern[x2] &= ~(1U << (x & 31));

	if (x > 1 && x < pw * 32 - 1 && y > 1 && y < ph - 1)
	{
		int n = GetBit(second, x - 1, y - 1, pw) + GetBit(second, x - 1, y, pw) + GetBit(second, x - 1, y + 1, pw) + GetBit(second, x, y - 1, pw) + GetBit(second, x, y + 1, pw) + GetBit(second, x + 1, y - 1, pw) + GetBit(second, x + 1, y, pw) + GetBit(second, x + 1, y + 1, pw);

		if ((GetBit(second, x, y, pw) == 1 && n == 2) || n == 3)
		{
			pattern[x2] |= 1U << (int)(x & 31);
		}
	}
	
	if (x < xoffset || x > (xoffset + 511) || y < yoffset || y > (yoffset + 511))
	{
		return;
	}

	int2 pos = (int2)((int)x - xoffset, (int)y - yoffset);
	int colour = GetBit(pattern, x, y, pw);
	float4 col = (float4)(colour, colour, colour, 1.0f);
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
	uint y = id / (pw * 32);
	uint x2 = x / 32 + y * pw;

	int bitWaarde = GetBit(pattern, x, y, pw);
	second[x2] |= bitWaarde << (int)(x & 31);
	//second[x2] = pattern[x2];
}